#include "sgemm.cuh"
#include <stdexcept>

#ifdef F16RAW
#include "convert.cuh"
#include "mma.cuh"
#endif

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

#ifdef F16RAW
// Convert only the current K32 operand tiles in LDS. No converted global buffers.
// threadIdx.x is the wave32 lane: GGML's fragment coordinates require x < 32.
template<int TileM, typename Packed, int TileN = 64>
__global__ __launch_bounds__(128) void halo_sgemm_wmma(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t sa, int64_t sb, int64_t sc) {
#if defined(AMD_WMMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    using tile_AB = tile<16, 8, Packed, get_input_data_layout()>;
    using tile_C = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
    static_assert(TileM == 64 || TileM == 128, "WMMA probe requires M64 or M128");
    static_assert(TileN == 64 || TileN == 128, "WMMA probe requires N64 or N128");
    static_assert(tile_AB::supported() && tile_C::supported(), "unsupported WMMA fragments");
    // 16 packed pairs cover K32; four padding pairs keep every row 16-byte aligned.
    constexpr int pitch = 20;
    constexpr int fragments_m = TileM / 32;
    constexpr int fragments_n = TileN / 32;
    __shared__ __align__(16) Packed aa[TileM][pitch];
    __shared__ __align__(16) Packed bb[TileN][pitch];
    const int tid = threadIdx.x + 32 * threadIdx.y;
    const int wave_row = (threadIdx.y % 2) * (TileM / 2);
    const int wave_col = (threadIdx.y / 2) * (TileN / 2);
    const int64_t row0 = int64_t(blockIdx.x) * TileM;
    const int64_t col0 = int64_t(blockIdx.y) * TileN;
    a += int64_t(blockIdx.z) * sa;
    b += int64_t(blockIdx.z) * sb;
    c += int64_t(blockIdx.z) * sc;
    tile_C acc[fragments_m][fragments_n];

    for (int64_t base = 0; base < k; base += 32) {
#pragma unroll
        for (int index = tid; index < TileM * 16; index += 128) {
            const int row = index / 16;
            const int64_t kk = base + 2 * (index % 16);
            const float lo = row0 + row < m && kk < k ? a[(row0 + row) * lda + kk] : 0.0f;
            const float hi = row0 + row < m && kk + 1 < k ? a[(row0 + row) * lda + kk + 1] : 0.0f;
            aa[row][index % 16] = ggml_cuda_cast<Packed>(make_float2(lo, hi));
        }
#pragma unroll
        for (int index = tid; index < TileN * 16; index += 128) {
            const int col = index / 16;
            const int64_t kk = base + 2 * (index % 16);
            const float lo = col0 + col < n && kk < k ? b[(col0 + col) * ldb + kk] : 0.0f;
            const float hi = col0 + col < n && kk + 1 < k ? b[(col0 + col) * ldb + kk + 1] : 0.0f;
            bb[col][index % 16] = ggml_cuda_cast<Packed>(make_float2(lo, hi));
        }
        __syncthreads();
#pragma unroll
        for (int kp = 0; kp < 16; kp += 8) {
            tile_AB a_frag[fragments_m];
            tile_AB b_frag[fragments_n];
#pragma unroll
            for (int i = 0; i < fragments_m; ++i) {
                load_ldmatrix(a_frag[i], &aa[wave_row + 16 * i][kp], pitch);
            }
#pragma unroll
            for (int j = 0; j < fragments_n; ++j) {
                load_ldmatrix(b_frag[j], &bb[wave_col + 16 * j][kp], pitch);
            }
#pragma unroll
            for (int i = 0; i < fragments_m; ++i) {
#pragma unroll
                for (int j = 0; j < fragments_n; ++j) {
                    mma(acc[i][j], a_frag[i], b_frag[j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < fragments_m; ++i) {
#pragma unroll
        for (int j = 0; j < fragments_n; ++j) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int64_t row = row0 + wave_row + 16 * i + tile_C::get_i(l);
                const int64_t col = col0 + wave_col + 16 * j + tile_C::get_j(l);
                if (row < m && col < n) {
                    c[col * ldc + row] = acc[i][j].x[l];
                }
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}
#endif

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
#ifdef F16RAW
        case 5:
            halo_sgemm_wmma<64, nv_bfloat162><<<dim3((m + 63) / 64, (n + 63) / 64, batches), dim3(32, 4), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 6:
            halo_sgemm_wmma<64, half2><<<dim3((m + 63) / 64, (n + 63) / 64, batches), dim3(32, 4), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 7:
            halo_sgemm_wmma<128, nv_bfloat162><<<dim3((m + 127) / 128, (n + 63) / 64, batches), dim3(32, 4), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 8:
            halo_sgemm_wmma<128, half2><<<dim3((m + 127) / 128, (n + 63) / 64, batches), dim3(32, 4), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 9:
            halo_sgemm_wmma<64, nv_bfloat162, 128><<<dim3((m + 63) / 64, (n + 127) / 128, batches), dim3(32, 4), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 10:
            halo_sgemm_wmma<64, half2, 128><<<dim3((m + 63) / 64, (n + 127) / 128, batches), dim3(32, 4), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
#endif
        default:
#ifdef F16RAW
            throw std::invalid_argument("SGEMM tile index must be 0..10");
#else
            throw std::invalid_argument("SGEMM tile index must be 0..4");
#endif
    }
}
