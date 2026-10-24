#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[1]) != "--log") return 2;
    const std::string mode = argv[2];
    ggml_time_init();
    const auto start = ggml_time_us();
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) throw std::runtime_error("GPU required");
    auto backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) throw std::runtime_error("GPU initialization failed");
    std::printf("backend=%s mode=%s\n", ggml_backend_name(backend), mode.c_str());
    std::fflush(stdout);
    const std::vector<int> sizes = mode == "repro" ? std::vector<int>{4096} :
        mode == "small" ? std::vector<int>{16, 1023, 1024} :
        std::vector<int>{1025, 2048, 4096, 4097, 32768, 65536};
    for (int cols : sizes) for (int rows : {1, 6}) for (auto order : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
        auto ctx = ggml_init({ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true});
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        auto output = ggml_argsort(ctx, input, order);
        auto graph = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(graph, output);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("GPU allocation failed");
        std::vector<float> data(cols * rows), after(data.size());
        std::vector<int> indices(data.size());
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
                float value = float((c * 109 + r * 17) % cols) - cols / 2.0f;
                if (pattern == 1) value = float(c % 7 - 3);
                if (pattern == 2 && c % 11 == 0) value = -std::numeric_limits<float>::infinity();
                if (pattern == 2 && c % 13 == 0) value = std::numeric_limits<float>::infinity();
                data[r * cols + c] = value;
            }
            ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
            for (int repeat = 0; repeat < 3; ++repeat) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("Graph execution failed");
                ggml_backend_tensor_get(output, indices.data(), 0, indices.size() * sizeof(int));
                ggml_backend_tensor_get(input, after.data(), 0, after.size() * sizeof(float));
                if (after != data) throw std::runtime_error("Argsort changed input logits");
                for (int r = 0; r < rows; ++r) {
                    std::vector<bool> seen(cols);
                    for (int c = 0; c < cols; ++c) {
                        const int index = indices[r * cols + c];
                        if (index < 0 || index >= cols || seen[index]) throw std::runtime_error("Invalid permutation");
                        seen[index] = true;
                        if (c) {
                            const float prev = data[r * cols + indices[r * cols + c - 1]];
                            const float next = data[r * cols + index];
                            if (order == GGML_SORT_ORDER_ASC ? prev > next : prev < next)
                                throw std::runtime_error("Incorrect sort order");
                        }
                    }
                }
            }
        }
        std::vector<double> times;
        for (int round = 0; round < 5; ++round) {
            const auto begin = ggml_time_us();
            for (int repeat = 0; repeat < 100; ++repeat) {
                if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("Timed graph execution failed");
            }
            ggml_backend_synchronize(backend);
            times.push_back((ggml_time_us() - begin) / 100.0);
        }
        std::sort(times.begin(), times.end());
        std::printf("argsort cols=%d rows=%d order=%s correct=true median_us=%.3f\n",
                    cols, rows, order == GGML_SORT_ORDER_ASC ? "asc" : "desc", times[2]);
        std::fflush(stdout);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::printf("session.wall_ms=%.3f\n", (ggml_time_us() - start) / 1000.0);
}
