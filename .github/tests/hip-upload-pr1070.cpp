#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[1]) != "--log") return 2;
    ggml_time_init();
    const auto started = ggml_time_us();
    std::vector<size_t> sizes = {16, 64, 65, 128, 256};
    sizes.insert(sizes.end(), 16, 128);
    constexpr size_t MiB = 1024 * 1024;
    std::vector<unsigned char> pattern(MiB);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = (i * 17 + i / 256) % 256;
    const std::string file = argv[2];
    if (!std::ifstream(file).good()) {
        auto * descriptors = ggml_init({ggml_tensor_overhead() * (sizes.size() + 1), nullptr, true});
        if (!descriptors) throw std::runtime_error("Cannot allocate tensor descriptors");
        auto * gguf = gguf_init_empty();
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto * tensor = ggml_new_tensor_1d(descriptors, GGML_TYPE_F32, sizes[i] * MiB / 4);
            ggml_set_name(tensor, ("weight_" + std::to_string(i)).c_str());
            gguf_add_tensor(gguf, tensor);
        }
        if (!gguf_write_to_file(gguf, file.c_str(), true)) throw std::runtime_error("GGUF metadata write failed");
        gguf_free(gguf);
        ggml_free(descriptors);
        std::ofstream output(file, std::ios::binary | std::ios::app);
        for (auto size : sizes) for (size_t i = 0; i < size; ++i) output.write(reinterpret_cast<const char *>(pattern.data()), pattern.size());
        output.close();
        if (!output) throw std::runtime_error("GGUF tensor payload write failed");
    }
    engine::core::BackendConfig config;
    config.type = engine::core::BackendType::Hip;
    engine::core::ExecutionContext execution(config);
    if (execution.backend_type() != engine::core::BackendType::Hip) throw std::runtime_error("HIP backend required");
    auto source = engine::assets::open_tensor_source(file);
    std::vector<ggml_context *> contexts;
    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<unsigned char> received(MiB);
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto * ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
        if (!ctx) throw std::runtime_error("Cannot allocate tensor context");
        auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sizes[i] * MiB / 4);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, execution.backend());
        if (!buffer || ggml_backend_buffer_is_host(buffer)) throw std::runtime_error("HIP device buffer required");
        contexts.push_back(ctx);
        buffers.push_back(buffer);
        std::cout << "upload.begin index=" << i << " bytes=" << ggml_nbytes(tensor) << std::endl;
        const auto begin = ggml_time_us();
        source->set_backend_tensor(tensor, "weight_" + std::to_string(i), engine::assets::TensorStorageType::Native,
                                   {static_cast<int64_t>(sizes[i] * MiB / 4)});
        ggml_backend_synchronize(execution.backend());
        const auto elapsed = ggml_time_us() - begin;
        for (size_t offset = 0; offset < ggml_nbytes(tensor); offset += MiB) {
            ggml_backend_tensor_get(tensor, received.data(), offset, received.size());
            if (received != pattern) throw std::runtime_error("Uploaded tensor bytes differ at index " + std::to_string(i));
        }
        std::cout << "upload.end index=" << i << " bytes=" << ggml_nbytes(tensor)
                  << " upload.wall_ms=" << elapsed / 1000.0 << " exact_bytes=true" << std::endl;
        std::ifstream status("/proc/self/status");
        for (std::string line; std::getline(status, line);) {
            if (line.rfind("RssFile:", 0) == 0 || line.rfind("RssAnon:", 0) == 0 || line.rfind("VmRSS:", 0) == 0)
                std::cout << line << std::endl;
        }
    }
    for (auto * buffer : buffers) ggml_backend_buffer_free(buffer);
    for (auto * ctx : contexts) ggml_free(ctx);
    std::cout << "session.wall_ms=" << (ggml_time_us() - started) / 1000.0 << std::endl;
}
