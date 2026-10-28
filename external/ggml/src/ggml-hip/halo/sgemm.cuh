#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

// Column-major C = A^T B, matching GGML's FP32 BLAS layout.
void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c, int64_t batches,
    hipStream_t stream);

// Raw-probe overload: tile_index 0..3 selects 64x64, 64x128, 128x64,
// or 128x128 (M x N). The original signature above always uses 64x64.
void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c, int64_t batches,
    hipStream_t stream, int tile_index);
