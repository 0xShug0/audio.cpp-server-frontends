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
template<int TileM, typename Packed, int TileN = 64,
         int Nwarps = (TileM == 128 && TileN == 128 ? 8 : 4), int SplitK = 1,
         bool AlignedLoads = false>
__global__ __launch_bounds__(32 * Nwarps) void halo_sgemm_wmma(
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
    constexpr int threads = 32 * Nwarps;
    constexpr int row_groups = Nwarps / 2;
    constexpr int fragments_m = TileM / (16 * row_groups);
    constexpr int fragments_n = TileN / 32;
    __shared__ __align__(16) Packed aa[TileM][pitch];
    __shared__ __align__(16) Packed bb[TileN][pitch];
    const int tid = threadIdx.x + 32 * threadIdx.y;
    const int wave_row = (threadIdx.y % row_groups) * (TileM / row_groups);
    const int wave_col = (threadIdx.y / row_groups) * (TileN / 2);
    const int64_t row0 = int64_t(blockIdx.x) * TileM;
    const int64_t col0 = int64_t(blockIdx.y) * TileN;
    const int64_t batch = int64_t(blockIdx.z) / SplitK;
    a += batch * sa;
    b += batch * sb;
    c += batch * sc;
    int64_t start_k = 0;
    int64_t end_k = k;
    if constexpr (SplitK > 1) {
        const int64_t slice = int64_t(blockIdx.z) % SplitK;
        // Round each slice up to K32 without adding to the full K dimension.
        const int64_t slice_k = (k / (SplitK * 32) + (k % (SplitK * 32) != 0)) * 32;
        const int64_t slice_start = slice * slice_k;
        start_k = slice_start < k ? slice_start : k;
        end_k = k - start_k < slice_k ? k : start_k + slice_k;
        c += slice * ldc * n;
    }
    tile_C acc[fragments_m][fragments_n];

    // Empty slices still store the zero-initialized accumulator plane below.
    for (int64_t base = start_k; base < end_k; base += 32) {
        if constexpr (AlignedLoads) {
            // Eligibility guarantees full K32 chunks and aligned rows/batches.
            // Each group loads eight F32 values, then writes four packed pairs.
#pragma unroll
            for (int i = 0; i < TileM * 4 / threads; ++i) {
                const int index = tid + i * threads;
                const int row = index / 4;
                const int pair = 4 * (index % 4);
                const int64_t kk = base + 2 * pair;
                float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (row0 + row < m) {
                    const float * src = a + (row0 + row) * lda + kk;
                    lo = *reinterpret_cast<const float4 *>(src);
                    hi = *reinterpret_cast<const float4 *>(src + 4);
                }
                aa[row][pair + 0] = ggml_cuda_cast<Packed>(make_float2(lo.x, lo.y));
                aa[row][pair + 1] = ggml_cuda_cast<Packed>(make_float2(lo.z, lo.w));
                aa[row][pair + 2] = ggml_cuda_cast<Packed>(make_float2(hi.x, hi.y));
                aa[row][pair + 3] = ggml_cuda_cast<Packed>(make_float2(hi.z, hi.w));
            }
#pragma unroll
            for (int i = 0; i < TileN * 4 / threads; ++i) {
                const int index = tid + i * threads;
                const int col = index / 4;
                const int pair = 4 * (index % 4);
                const int64_t kk = base + 2 * pair;
                float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (col0 + col < n) {
                    const float * src = b + (col0 + col) * ldb + kk;
                    lo = *reinterpret_cast<const float4 *>(src);
                    hi = *reinterpret_cast<const float4 *>(src + 4);
                }
                bb[col][pair + 0] = ggml_cuda_cast<Packed>(make_float2(lo.x, lo.y));
                bb[col][pair + 1] = ggml_cuda_cast<Packed>(make_float2(lo.z, lo.w));
                bb[col][pair + 2] = ggml_cuda_cast<Packed>(make_float2(hi.x, hi.y));
                bb[col][pair + 3] = ggml_cuda_cast<Packed>(make_float2(hi.z, hi.w));
            }
        } else {
#pragma unroll
            for (int index = tid; index < TileM * 16; index += threads) {
                const int row = index / 16;
                const int64_t kk = base + 2 * (index % 16);
                const float lo = row0 + row < m && kk < end_k ? a[(row0 + row) * lda + kk] : 0.0f;
                const float hi = row0 + row < m && kk + 1 < end_k ? a[(row0 + row) * lda + kk + 1] : 0.0f;
                aa[row][index % 16] = ggml_cuda_cast<Packed>(make_float2(lo, hi));
            }
#pragma unroll
            for (int index = tid; index < TileN * 16; index += threads) {
                const int col = index / 16;
                const int64_t kk = base + 2 * (index % 16);
                const float lo = col0 + col < n && kk < end_k ? b[(col0 + col) * ldb + kk] : 0.0f;
                const float hi = col0 + col < n && kk + 1 < end_k ? b[(col0 + col) * ldb + kk + 1] : 0.0f;
                bb[col][index % 16] = ggml_cuda_cast<Packed>(make_float2(lo, hi));
            }
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
    int64_t sa, int64_t sb, int64_t sc, int64_t batches, hipStream_t stream, int tile_index
#ifdef F16RAW
    , const char ** route
#endif
    ) {
#ifdef F16RAW
    if (route != nullptr) {
        *route = nullptr;
    }
    if (tile_index >= 13 && tile_index <= 18 && batches != 1) {
        throw std::invalid_argument("split-K SGEMM probes require batches == 1");
    }
#endif
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
        case 11:
            halo_sgemm_wmma<128, nv_bfloat162, 128><<<dim3((m + 127) / 128, (n + 127) / 128, batches), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 12:
            halo_sgemm_wmma<128, half2, 128><<<dim3((m + 127) / 128, (n + 127) / 128, batches), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 13:
            halo_sgemm_wmma<128, nv_bfloat162, 128, 8, 2><<<dim3((m + 127) / 128, (n + 127) / 128, 2), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 14:
            halo_sgemm_wmma<128, half2, 128, 8, 2><<<dim3((m + 127) / 128, (n + 127) / 128, 2), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 15:
            halo_sgemm_wmma<128, nv_bfloat162, 128, 8, 4><<<dim3((m + 127) / 128, (n + 127) / 128, 4), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 16:
            halo_sgemm_wmma<128, half2, 128, 8, 4><<<dim3((m + 127) / 128, (n + 127) / 128, 4), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 17:
            halo_sgemm_wmma<128, nv_bfloat162, 128, 8, 8><<<dim3((m + 127) / 128, (n + 127) / 128, 8), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 18:
            halo_sgemm_wmma<128, half2, 128, 8, 8><<<dim3((m + 127) / 128, (n + 127) / 128, 8), dim3(32, 8), 0, stream>>>(
                a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
            break;
        case 19:
        case 20: {
            const bool aligned = reinterpret_cast<std::uintptr_t>(a) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(b) % 16 == 0 &&
                lda % 4 == 0 && ldb % 4 == 0 && k % 32 == 0 &&
                (batches <= 1 || (sa % 4 == 0 && sb % 4 == 0));
            if (route != nullptr) {
                *route = aligned ? "aligned_f32x8" : "wide_guarded";
            }
            const dim3 grid((m + 127) / 128, (n + 127) / 128, batches);
            if (tile_index == 19) {
                if (aligned) {
                    halo_sgemm_wmma<128, nv_bfloat162, 128, 8, 1, true><<<grid, dim3(32, 8), 0, stream>>>(
                        a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
                } else {
                    halo_sgemm_wmma<128, nv_bfloat162, 128><<<grid, dim3(32, 8), 0, stream>>>(
                        a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
                }
            } else {
                if (aligned) {
                    halo_sgemm_wmma<128, half2, 128, 8, 1, true><<<grid, dim3(32, 8), 0, stream>>>(
                        a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
                } else {
                    halo_sgemm_wmma<128, half2, 128><<<grid, dim3(32, 8), 0, stream>>>(
                        a, b, c, m, n, k, lda, ldb, ldc, sa, sb, sc);
                }
            }
            break;
        }
#endif
        default:
#ifdef F16RAW
            throw std::invalid_argument("SGEMM tile index must be 0..20");
#else
            throw std::invalid_argument("SGEMM tile index must be 0..4");
#endif
    }
}
