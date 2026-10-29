#include "halo.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/dequantize.cuh"
#include "ggml-cuda/mma.cuh"
#include <climits>

namespace {

template<ggml_type type>
__device__ __forceinline__ float load_weight(const char * row, int64_t k) {
    if constexpr (type == GGML_TYPE_F16) {
        return __half2float(reinterpret_cast<const half *>(row)[k]);
    } else if constexpr (type == GGML_TYPE_BF16) {
        return ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(row)[k]);
    } else {
        float2 pair;
        if constexpr (type == GGML_TYPE_Q4_0) {
            dequantize_q4_0(row, k / QK4_0, k % (QK4_0 / 2), pair);
            return k % QK4_0 < QK4_0 / 2 ? pair.x : pair.y;
        } else {
            dequantize_q8_0(row, k / QK8_0, (k % QK8_0) & ~1, pair);
            return k & 1 ? pair.y : pair.x;
        }
    }
}

template<ggml_type type>
__global__ __launch_bounds__(128) void halo_matvec(
        const char * a, const float * b, float * c,
        int64_t m, int64_t n, int64_t k, size_t lda, int64_t ldb) {
    const int lane = threadIdx.x % 32;
    const int64_t row0 = int64_t(blockIdx.x) * 16 + (threadIdx.x / 32) * 4;
    const int64_t col = blockIdx.y;
    float sum[4] = {};
    for (int64_t kk = lane; kk < k; kk += 32) {
        const float v = b[col * ldb + kk];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (row0 + r < m) {
                sum[r] = fmaf(load_weight<type>(a + (row0 + r) * lda, kk), v, sum[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float v = warp_reduce_sum<32>(sum[r]);
        if (lane == 0 && row0 + r < m) {
            c[col * m + row0 + r] = v;
        }
    }
}

template<ggml_type type, typename pair_type>
__global__ __launch_bounds__(128) void halo_matmul_tiled(
        const char * a, const float * b, float * c,
        int64_t m, int64_t n, int64_t k, size_t lda, int64_t ldb) {
#if defined(AMD_WMMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    using tile_AB = tile<16, 8, pair_type, get_input_data_layout()>;
    using tile_C = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
    constexpr int pitch = 20;
    __shared__ __align__(16) pair_type aa[64][pitch];
    __shared__ __align__(16) pair_type bb[64][pitch];
    const int tid = threadIdx.x;
    const int wave_row = (tid / 32 % 2) * 32;
    const int wave_col = (tid / 64) * 32;
    const int64_t row0 = int64_t(blockIdx.x) * 64;
    const int64_t col0 = int64_t(blockIdx.y) * 64;
    tile_C acc[2][2];
    for (int64_t base = 0; base < k; base += 32) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int index = tid + i * 128;
            const int row = index / 16;
            const int pair = index % 16;
            const int64_t kk = base + 2 * pair;
            float2 av = make_float2(0, 0), bv = make_float2(0, 0);
            if (row0 + row < m) {
                const char * src = a + (row0 + row) * lda;
                av.x = load_weight<type>(src, kk);
                av.y = load_weight<type>(src, kk + 1);
            }
            if (col0 + row < n) {
                const float * src = b + (col0 + row) * ldb + kk;
                bv = make_float2(src[0], src[1]);
            }
            aa[row][pair] = ggml_cuda_cast<pair_type>(av);
            bb[row][pair] = ggml_cuda_cast<pair_type>(bv);
        }
        __syncthreads();
#pragma unroll
        for (int kp = 0; kp < 16; kp += 8) {
            tile_AB af[2], bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                load_ldmatrix(af[i], &aa[wave_row + 16 * i][kp], pitch);
                load_ldmatrix(bf[i], &bb[wave_col + 16 * i][kp], pitch);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    mma(acc[i][j], af[i], bf[j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int64_t row = row0 + wave_row + 16 * i + tile_C::get_i(l);
                const int64_t col = col0 + wave_col + 16 * j + tile_C::get_j(l);
                if (row < m && col < n) {
                    c[col * m + row] = acc[i][j].x[l];
                }
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}

template<ggml_type type>
void launch(const ggml_tensor * a, const ggml_tensor * b, ggml_tensor * out, hipStream_t stream) {
    const int64_t m = a->ne[1], n = b->ne[1], k = a->ne[0];
    if (n <= 8) {
        halo_matvec<type><<<dim3((m + 15) / 16, n), 128, 0, stream>>>(
            static_cast<const char *>(a->data), static_cast<const float *>(b->data),
            static_cast<float *>(out->data), m, n, k, a->nb[1], b->nb[1] / sizeof(float));
    } else {
        using pair_type = std::conditional_t<type == GGML_TYPE_BF16, nv_bfloat162, half2>;
        halo_matmul_tiled<type, pair_type><<<dim3((m + 63) / 64, (n + 63) / 64), 128, 0, stream>>>(
            static_cast<const char *>(a->data), static_cast<const float *>(b->data),
            static_cast<float *>(out->data), m, n, k, a->nb[1], b->nb[1] / sizeof(float));
    }
}

} // namespace

bool ggml_hip_halo_mul_mat_typed(const ggml_tensor * a, const ggml_tensor * b,
                                ggml_tensor * out, hipStream_t stream) {
    if (out->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(out, 1) != GGML_MUL_MAT_LOWERING_DEFAULT ||
        b->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
        b->nb[0] != sizeof(float) || !ggml_is_contiguous(out)) {
        return false;
    }
    if (a->type != GGML_TYPE_F16 && a->type != GGML_TYPE_BF16 &&
        a->type != GGML_TYPE_Q8_0 && a->type != GGML_TYPE_Q4_0) {
        return false;
    }
    const int64_t m = a->ne[1], n = b->ne[1], k = a->ne[0];
    if (m < 16 || m > INT_MAX || n < 1 || n > 65535 * 64 || k < 32 || k > INT_MAX ||
        k % 32 != 0 || a->nb[0] != ggml_type_size(a->type) ||
        a->nb[1] < ggml_row_size(a->type, k) ||
        b->nb[1] < size_t(k) * sizeof(float) || b->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (n > 8 && (n < 32 || ggml_get_op_params_i32(out, 0) == GGML_PREC_F32)) {
        return false;
    }
    switch (a->type) {
        case GGML_TYPE_F16:  launch<GGML_TYPE_F16>(a, b, out, stream); break;
        case GGML_TYPE_BF16: launch<GGML_TYPE_BF16>(a, b, out, stream); break;
        case GGML_TYPE_Q8_0: launch<GGML_TYPE_Q8_0>(a, b, out, stream); break;
        case GGML_TYPE_Q4_0: launch<GGML_TYPE_Q4_0>(a, b, out, stream); break;
        default: GGML_ABORT("unreachable Halo matmul type");
    }
    return true;
}
