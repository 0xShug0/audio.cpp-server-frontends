#pragma once

#include "../implementation.cuh"

bool ggml_hip_halo_mul_mat(const ggml_tensor *, const ggml_tensor *, ggml_tensor *, hipStream_t);
bool ggml_hip_halo_mul_mat_typed(const ggml_tensor *, const ggml_tensor *, ggml_tensor *, hipStream_t);
bool ggml_hip_halo_im2col(const ggml_tensor *, ggml_tensor *, hipStream_t);

bool ggml_hip_halo_cpy(const ggml_tensor *, ggml_tensor *, hipStream_t);
