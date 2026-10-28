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
// Index 4 uses 64x64 with aligned float4 LDS reads and contiguous 4x4 thread outputs.
// F16RAW only: 5=BF16 64x64, 6=F16 64x64, 7=BF16 128x64, 8=F16 128x64.
// These explicit default-precision probes convert F32 inputs in LDS and use
// GGML's AMD WMMA helpers with F32 accumulation/output and no global scratch.
// Keep explicit-F32 requests on the original overload or indices 0..4.
void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c, int64_t batches,
    hipStream_t stream, int tile_index);
