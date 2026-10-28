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
// Indices 9=BF16 64x128 and 10=F16 64x128 reuse A across twice as many columns.
// Indices 11=BF16 128x128 and 12=F16 128x128 use eight wave32 groups (256 threads),
// four wave rows by two wave columns: 32x64 per wave, eight accumulator fragments.
// Indices 5..12 convert F32 inputs in LDS and use GGML's AMD WMMA helpers
// with F32 accumulation/output and no global scratch.
// Split-K wide probes: 13=BF16/14=F16 split2, 15=BF16/16=F16 split4,
// 17=BF16/18=F16 split8. These require batches == 1 and write partials only.
// Caller supplies C storage for SplitK planes of ldc*n floats (packed: ldc=m).
// Plane s is C + s*ldc*n; each K slice has ceil(k/(SplitK*32))*32 elements,
// clamped to K. Empty slices write zeros. A/B retain their full-K row strides.
// Reduce planes with existing GGML add operations on the same HIP context stream;
// these probes perform no atomics, allocation, or reduction.
// Keep explicit-F32 requests on the original overload or indices 0..4.
void ggml_hip_halo_sgemm(const float * a, const float * b, float * c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c, int64_t batches,
    hipStream_t stream, int tile_index);
