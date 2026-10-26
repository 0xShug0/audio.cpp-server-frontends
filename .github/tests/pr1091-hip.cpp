#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class Pattern { Norm, Mul, MulAdd, MulRound, AddRound, SwiGluRound };

static float bf16(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

static void run_case(ggml_backend_t backend, std::ofstream & file,
                     int width, int rows, int channels, int samples,
                     bool strided, Pattern pattern) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("ggml_init failed");
    const int pitch = strided ? 2 * width : width;
    const int count = width * rows * channels * samples;
    auto * storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, pitch, rows, channels, samples);
    auto * x = strided ? ggml_view_4d(ctx, storage, width, rows, channels, samples,
                                     storage->nb[1], storage->nb[2], storage->nb[3], 0) : storage;
    auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    ggml_set_input(storage);
    ggml_set_input(weight);
    ggml_set_input(bias);
    std::vector<float> data(static_cast<size_t>(pitch * rows * channels * samples));
    std::vector<float> w(static_cast<size_t>(width)), b(w.size());
    for (size_t i = 0; i < data.size(); ++i) data[i] = 3.0f * std::sin(float(i) * 0.173f + 0.3f);
    for (int i = 0; i < width; ++i) {
        w[i] = 0.25f + 0.7f * std::cos(float(i) * 0.33f);
        b[i] = 0.4f * std::sin(float(i) * 0.91f);
    }
    constexpr float eps = 1e-6f;
    ggml_tensor * out = nullptr;
    if (pattern == Pattern::AddRound) {
        out = ggml_round_bf16(ctx, ggml_add(ctx, x, bias));
    } else if (pattern == Pattern::SwiGluRound) {
        auto * silu = ggml_round_bf16(ctx, ggml_silu(ctx, ggml_round_bf16(ctx, x)));
        out = ggml_round_bf16(ctx, ggml_mul(ctx, silu, ggml_round_bf16(ctx, weight)));
    } else {
        out = ggml_rms_norm(ctx, x, eps);
        if (pattern != Pattern::Norm) out = ggml_mul(ctx, out, weight);
        if (pattern == Pattern::MulAdd) out = ggml_add(ctx, out, bias);
        if (pattern == Pattern::MulRound) out = ggml_round_bf16(ctx, out);
    }
    ggml_set_output(out);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("GPU allocation failed");
    std::vector<float> result(static_cast<size_t>(count));
    for (int repeat = 0; repeat < 3; ++repeat) {
        ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(float));
        ggml_backend_tensor_set(weight, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(bias, b.data(), 0, b.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph compute failed");
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        for (int row = 0; row < rows * channels * samples; ++row) {
            double squares = 0;
            for (int col = 0; col < width; ++col) {
                double value = data[row * pitch + col];
                squares += value * value;
            }
            const float scale = float(1.0 / std::sqrt(squares / width + eps));
            for (int col = 0; col < width; ++col) {
                const float value = data[row * pitch + col];
                float expected;
                if (pattern == Pattern::AddRound) {
                    expected = bf16(value + b[col]);
                } else if (pattern == Pattern::SwiGluRound) {
                    const float gate = bf16(value);
                    expected = bf16(bf16(gate / (1.0f + std::exp(-gate))) * bf16(w[col]));
                } else {
                    expected = scale * value;
                    if (pattern != Pattern::Norm) expected *= w[col];
                    if (pattern == Pattern::MulAdd) expected += b[col];
                    if (pattern == Pattern::MulRound) expected = bf16(expected);
                }
                const float actual = result[row * width + col];
                // GPU reductions and exp differ slightly from the independent scalar reference.
                // Cross-revision comparison below is exact, not tolerance-based.
                const float tolerance = pattern == Pattern::MulRound || pattern == Pattern::SwiGluRound
                    ? 0.01f * std::max(1.0f, std::fabs(expected))
                    : 2e-5f * std::max(1.0f, std::fabs(expected));
                if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
                    std::fprintf(stderr, "scalar mismatch pattern=%d width=%d row=%d col=%d actual=%.9g expected=%.9g\n",
                                 int(pattern), width, row, col, actual, expected);
                    throw std::runtime_error("independent scalar reference mismatch");
                }
            }
        }
        file.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
        if (!file) throw std::runtime_error("output write failed");
    }
    std::printf("PASS pattern=%d width=%d rows=%d channels=%d samples=%d strided=%d repeats=3\n",
                int(pattern), width, rows, channels, samples, int(strided));
    std::fflush(stdout);
    auto device = ggml_backend_get_device(backend);
    auto clear = reinterpret_cast<void (*)(ggml_backend_t, const ggml_cgraph *)>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_cuda_clear_graph"));
    if (clear) clear(backend, graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) throw std::runtime_error("AMD GPU required");
        auto backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) throw std::runtime_error("GPU initialization failed");
        const std::string name = ggml_backend_name(backend);
        if (name.find("ROCm") == std::string::npos && name.find("HIP") == std::string::npos)
            throw std::runtime_error("HIP backend required, got " + name);
        std::printf("backend=%s; default options; no opt-in hook called\n", name.c_str());
        std::ofstream file(argv[1], std::ios::binary);
        if (!file) throw std::runtime_error("cannot open output");
        int cases = 0;
        for (int width : {1, 17, 64, 1023, 1024, 2048, 8192}) {
            for (int shape = 0; shape < 3; ++shape) {
                for (bool strided : {false, true}) {
                    for (auto pattern : {Pattern::Norm, Pattern::Mul, Pattern::MulAdd,
                                         Pattern::MulRound, Pattern::AddRound, Pattern::SwiGluRound}) {
                        run_case(backend, file, width, shape == 0 ? 1 : 4,
                                 shape == 2 ? 2 : 1, shape == 2 ? 2 : 1, strided, pattern);
                        ++cases;
                    }
                }
            }
        }
        ggml_backend_free(backend);
        std::printf("PASS cases=%d\n", cases);
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
