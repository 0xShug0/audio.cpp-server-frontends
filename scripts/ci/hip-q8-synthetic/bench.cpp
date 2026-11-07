#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Graph {
    ggml_backend_t backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_cgraph * graph;
    ggml_tensor * output;

    explicit Graph(ggml_backend_t be)
        : backend(be), ctx(ggml_init({1024 * 1024, nullptr, true})),
          buffer(nullptr), graph(ggml_new_graph(ctx)), output(nullptr) {}

    Graph(ggml_backend_t be, int k, int m, int n,
          const std::vector<unsigned char> & weights, const std::vector<float> & input)
        : Graph(be) {
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
        auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        output = ggml_mul_mat(ctx, a, b);
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "graph allocation failed");
        ggml_backend_tensor_set(a, weights.data(), 0, weights.size());
        ggml_backend_tensor_set(b, input.data(), 0, input.size() * sizeof(float));
    }
    ~Graph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    void run() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "HIP graph failed");
    }
    std::vector<float> read() {
        std::vector<float> values(ggml_nelements(output));
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
    double time() {
        constexpr int repeats = 64;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) run();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
    }
};

static double median(std::vector<double> values);

static void projection_blocks(ggml_backend_t a, ggml_backend_t b, int hidden, int kv,
                              int ff, int cols, std::mt19937 & rng) {
    Graph ga(a), gb(b);
    std::normal_distribution<float> sample(0, 0.1f);
    struct Upload {
        ggml_tensor * a;
        ggml_tensor * b;
        std::vector<unsigned char> bytes;
    };
    std::vector<Upload> uploads;
    // Distinct weights across four layers avoid timing one permanently hot matrix.
    for (int layer = 0; layer < 4; ++layer) {
        const int shapes[][2] = {{hidden, hidden}, {hidden, kv}, {hidden, kv},
                                {hidden, hidden}, {hidden, ff}, {hidden, ff}, {ff, hidden}};
        for (const auto & shape : shapes) {
            const int k = shape[0], m = shape[1];
            std::vector<float> source(size_t(k) * m);
            for (auto & value : source) value = sample(rng);
            std::vector<unsigned char> weights(ggml_row_size(GGML_TYPE_Q8_0, k) * m);
            require(ggml_quantize_chunk(GGML_TYPE_Q8_0, source.data(), weights.data(), 0, m, k, nullptr)
                    == weights.size(), "block quantization size mismatch");
            auto * wa = ggml_new_tensor_2d(ga.ctx, GGML_TYPE_Q8_0, k, m);
            auto * wb = ggml_new_tensor_2d(gb.ctx, GGML_TYPE_Q8_0, k, m);
            auto * xa = ggml_new_tensor_2d(ga.ctx, GGML_TYPE_F32, k, cols);
            auto * xb = ggml_new_tensor_2d(gb.ctx, GGML_TYPE_F32, k, cols);
            uploads.push_back({wa, wb, std::move(weights)});
            std::vector<float> values(k * cols);
            for (auto & value : values) value = sample(rng);
            std::vector<unsigned char> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            uploads.push_back({xa, xb, std::move(bytes)});
            ga.output = ggml_mul_mat(ga.ctx, wa, xa);
            gb.output = ggml_mul_mat(gb.ctx, wb, xb);
            ggml_build_forward_expand(ga.graph, ga.output);
            ggml_build_forward_expand(gb.graph, gb.output);
        }
    }
    ga.buffer = ggml_backend_alloc_ctx_tensors(ga.ctx, a);
    gb.buffer = ggml_backend_alloc_ctx_tensors(gb.ctx, b);
    require(ga.buffer && gb.buffer, "projection block allocation failed");
    for (const auto & item : uploads) {
        ggml_backend_tensor_set(item.a, item.bytes.data(), 0, item.bytes.size());
        ggml_backend_tensor_set(item.b, item.bytes.data(), 0, item.bytes.size());
    }
    for (int i = 0; i < 8; ++i) { ga.run(); gb.run(); }
    for (int i = 0; i < ggml_graph_n_nodes(ga.graph); ++i) {
        ga.output = ggml_graph_node(ga.graph, i);
        gb.output = ggml_graph_node(gb.graph, i);
        const auto va = ga.read(), vb = gb.read();
        double error = 0, norm = 0;
        for (size_t j = 0; j < va.size(); ++j) {
            const double delta = double(va[j]) - vb[j];
            error += delta * delta;
            norm += double(va[j]) * va[j];
        }
        require(norm > 0 && std::sqrt(error / norm) <= 1e-5, "projection block parity failed");
    }
    std::vector<double> ta, tb;
    for (int i = 0; i < 3; ++i) {
        ta.push_back(ga.time()); tb.push_back(gb.time());
        tb.push_back(gb.time()); ta.push_back(ga.time());
    }
    const double ma = median(ta), mb = median(tb);
    std::printf("projection_blocks,H=%d,KV=%d,FF=%d,N=%d,default_us=%.3f,selected_us=%.3f,reduction_pct=%.2f\n",
                hidden,kv,ff,cols,ma,mb,100*(1-mb/ma));
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size()-1)/2] + values[values.size()/2]) / 2;
}

int main() {
    auto * dev = ggml_backend_dev_by_name("ROCm0");
    require(dev != nullptr, "ROCm0 unavailable");
    auto * a = ggml_backend_dev_init(dev, nullptr);
    auto * b = ggml_backend_dev_init_with_impl(dev, "gfx1151_v1", nullptr);
    require(a && b, "HIP backend initialization failed");
    std::printf("A=%s B=%s\n", ggml_backend_get_impl(a), ggml_backend_get_impl(b));
    std::puts("K,M,N,default_us,selected_us,latency_reduction_pct,max_abs_error,relative_l2");
    std::mt19937 rng(1234);
    std::normal_distribution<float> sample(0, 0.1f);
    // YuE2 CFG projection dimensions plus small/large TTS decoder projections.
    const int shapes[][2] = {{2048,1024}, {2048,2048}, {2048,6144}, {6144,2048},
                            {1536,1536}, {1536,8960}, {8960,1536}, {2048,32769}};
    for (const auto & shape : shapes) {
        const int k = shape[0], m = shape[1];
        std::vector<float> source(size_t(k) * m);
        for (auto & value : source) value = sample(rng);
        std::vector<unsigned char> weights(ggml_row_size(GGML_TYPE_Q8_0, k) * m);
        require(ggml_quantize_chunk(GGML_TYPE_Q8_0, source.data(), weights.data(), 0, m, k, nullptr)
                == weights.size(), "quantization size mismatch");
        for (int n : {1, 2, 4, 8}) {
            std::vector<float> input(k*n);
            for (auto & value : input) value = sample(rng);
            Graph ga(a,k,m,n,weights,input), gb(b,k,m,n,weights,input);
            for (int i = 0; i < 8; ++i) { ga.run(); gb.run(); }
            const auto va = ga.read(), vb = gb.read();
            double max_error = 0, error_sq = 0, ref_sq = 0;
            for (size_t i = 0; i < va.size(); ++i) {
                require(std::isfinite(va[i]) && std::isfinite(vb[i]), "nonfinite output");
                const double error = double(va[i]) - vb[i];
                max_error = std::max(max_error, std::abs(error));
                error_sq += error * error;
                ref_sq += double(va[i]) * va[i];
            }
            require(ref_sq > 0 && std::sqrt(error_sq / ref_sq) <= 1e-5, "HIP parity failed");
            std::vector<double> ta, tb;
            for (int i = 0; i < 3; ++i) {
                ta.push_back(ga.time());
                tb.push_back(gb.time());
                tb.push_back(gb.time());
                ta.push_back(ga.time());
            }
            const double ma = median(ta), mb = median(tb);
            std::printf("%d,%d,%d,%.3f,%.3f,%.2f,%.8g,%.8g\n", k,m,n,ma,mb,
                        100*(1-mb/ma),max_error,std::sqrt(error_sq/ref_sq));
            std::fflush(stdout);
        }
    }
    projection_blocks(a, b, 2048, 1024, 6144, 2, rng);
    projection_blocks(a, b, 1536, 256, 8960, 1, rng);
    ggml_backend_free(b);
    ggml_backend_free(a);
}
