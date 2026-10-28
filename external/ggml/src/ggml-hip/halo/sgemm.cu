#include "sgemm.cuh"
#include <stdexcept>

namespace {

// Reuse each operand tile across output rows/columns. Unlike reduced-
// precision matrix instructions, this retains FP32 inputs and accumulation.
template<int TileM, int TileN, bool VectorLds = false>
__global__ __launch_bounds__(256) void halo_sgemm(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc) {
    static_assert(!VectorLds || (TileM == 64 && TileN == 64), "vector LDS requires a 64x64 tile");
    __shared__ __align__(VectorLds ? 16 : alignof(float)) float aa[32][VectorLds ? 68 : TileM + 1];
    __shared__ __align__(VectorLds ? 16 : alignof(float)) float bb[32][VectorLds ? 68 : TileN + 1];
    const int tid = threadIdx.x;
    const int row = (tid % 16) * (VectorLds ? 4 : 1);
    const int col = (tid / 16) * (VectorLds ? 4 : 1);
    const int64_t row0 = int64_t(blockIdx.x) * TileM;
    const int64_t col0 = int64_t(blockIdx.y) * TileN;
    a += int64_t(blockIdx.z) * sa;
    b += int64_t(blockIdx.z) * sb;
    c += int64_t(blockIdx.z) * sc;
    float sum[TileM / 16][TileN / 16] = {};
    for (int64_t base = 0; base < k; base += 32) {
#pragma unroll
        for (int i = 0; i < (TileM > TileN ? TileM : TileN) * 32 / 256; ++i) {
            const int index = tid + i * 256;
            const int kk = index % 32;
            const int rr = index / 32;
            if (i < TileM * 32 / 256) {
                aa[kk][rr] = base + kk < k && row0 + rr < m ? a[(row0 + rr) * lda + base + kk] : 0;
            }
            if (i < TileN * 32 / 256) {
                bb[kk][rr] = base + kk < k && col0 + rr < n ? b[(col0 + rr) * ldb + base + kk] : 0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            float av[TileM / 16], bv[TileN / 16];
            if constexpr (VectorLds) {
                const float4 av4 = *reinterpret_cast<const float4 *>(&aa[kk][row]);
                const float4 bv4 = *reinterpret_cast<const float4 *>(&bb[kk][col]);
                av[0] = av4.x; av[1] = av4.y; av[2] = av4.z; av[3] = av4.w;
                bv[0] = bv4.x; bv[1] = bv4.y; bv[2] = bv4.z; bv[3] = bv4.w;
            } else {
#pragma unroll
                for (int i = 0; i < (TileM > TileN ? TileM : TileN) / 16; ++i) {
                    if (i < TileM / 16) av[i] = aa[kk][row + i * 16];
                    if (i < TileN / 16) bv[i] = bb[kk][col + i * 16];
                }
            }
#pragma unroll
            for (int i = 0; i < TileM / 16; ++i) {
#pragma unroll
                for (int j = 0; j < TileN / 16; ++j) {
                    sum[i][j] = fmaf(av[i], bv[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TileM / 16; ++i) {
#pragma unroll
        for (int j = 0; j < TileN / 16; ++j) {
            if (row0 + row + i * (VectorLds ? 1 : 16) < m &&
                col0 + col + j * (VectorLds ? 1 : 16) < n) {
                c[(col0 + col + j * (VectorLds ? 1 : 16)) * ldc +
                    row0 + row + i * (VectorLds ? 1 : 16)] = sum[i][j];
            }
        }
    }
}

} // namespace

void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc, int64_t batches, hipStream_t stream) {
    halo_sgemm<64, 64><<<dim3((m + 63) / 64, (n + 63) / 64, batches), 256, 0, stream>>>(
        a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
}

void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc, int64_t batches, hipStream_t stream, int tile_index) {
    switch (tile_index) {
        case 0:
            halo_sgemm<64, 64><<<dim3((m + 63) / 64, (n + 63) / 64, batches), 256, 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 1:
            halo_sgemm<64, 128><<<dim3((m + 63) / 64, (n + 127) / 128, batches), 256, 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 2:
            halo_sgemm<128, 64><<<dim3((m + 127) / 128, (n + 63) / 64, batches), 256, 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 3:
            halo_sgemm<128, 128><<<dim3((m + 127) / 128, (n + 127) / 128, batches), 256, 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 4:
            halo_sgemm<64, 64, true><<<dim3((m + 63) / 64, (n + 63) / 64, batches), 256, 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        default:
            throw std::invalid_argument("SGEMM tile index must be 0..4");
    }
}
