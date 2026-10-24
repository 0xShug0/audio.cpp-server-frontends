#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr size_t mib = 1024 * 1024;
constexpr size_t file_bytes = 256 * mib;
// A large tensor, a smaller-than-chunk tensor, and a small tail. Leave room
// for GGUF metadata so the entire synthetic file stays exactly 256 MiB.
const std::vector<size_t> sizes = {192 * mib + 4, 63 * mib - 4, mib - 4096};

uint32_t expected(size_t tensor, size_t word) {
    return 0x3f000000U | ((uint32_t(word) * 2654435761U ^ uint32_t(tensor) * 73244475U) & 0x007fffffU);
}

void rss(const std::string & label) {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::cout << "MEM " << label;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0 || line.rfind("RssFile:", 0) == 0 || line.rfind("RssAnon:", 0) == 0) {
            std::cout << " | " << line;
        }
    }
    std::cout << '\n';
}

void generate(const std::string & path) {
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    gguf_context * gguf = gguf_init_empty();
    if (!ctx || !gguf) throw std::runtime_error("metadata allocation failed");
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sizes[i] / sizeof(uint32_t));
        ggml_set_name(tensor, ("weight_" + std::to_string(i)).c_str());
        gguf_add_tensor(gguf, tensor);
    }
    if (!gguf_write_to_file(gguf, path.c_str(), true)) throw std::runtime_error("GGUF metadata write failed");
    std::ofstream out(path, std::ios::binary | std::ios::app);
    std::vector<uint32_t> chunk(mib / sizeof(uint32_t));
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto target = gguf_get_meta_size(gguf) + gguf_get_tensor_offset(gguf, i);
        if (size_t(out.tellp()) > target) throw std::runtime_error("bad GGUF offset");
        while (size_t(out.tellp()) < target) out.put(0);
        for (size_t offset = 0; offset < sizes[i] / 4; offset += chunk.size()) {
            const size_t count = std::min(chunk.size(), sizes[i] / 4 - offset);
            for (size_t j = 0; j < count; ++j) chunk[j] = expected(i, offset + j);
            out.write(reinterpret_cast<const char *>(chunk.data()), count * sizeof(uint32_t));
        }
    }
    if (!out || size_t(out.tellp()) > file_bytes) throw std::runtime_error("synthetic GGUF exceeds 256 MiB");
    out.close();
    std::filesystem::resize_file(path, file_bytes);
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open generated GGUF");
    fsync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    gguf_free(gguf);
    ggml_free(ctx);
    std::cout << "FILE bytes=" << std::filesystem::file_size(path) << '\n';
}

void upload(const std::string & path, bool cpu) {
    using namespace engine;
    core::BackendConfig config;
    config.type = cpu ? core::BackendType::Cpu : core::BackendType::Hip;
    ggml_backend_t backend = core::init_backend(config);
    if (!backend) throw std::runtime_error("backend initialization failed");
    std::cout << "BACKEND " << ggml_backend_name(backend) << '\n';
    auto source = assets::open_tensor_source(path);
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("tensor context allocation failed");
    std::vector<ggml_tensor *> tensors;
    for (size_t i = 0; i < sizes.size(); ++i) {
        tensors.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sizes[i] / 4));
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("device tensor allocation failed");
    std::cout << "BUFFER host=" << ggml_backend_buffer_is_host(buffer)
              << " name=" << ggml_backend_buffer_name(buffer) << '\n';
    if (!cpu && ggml_backend_buffer_is_host(buffer)) throw std::runtime_error("HIP test selected a host buffer");
    std::vector<uint32_t> readback(mib / 4);
    rss("before_upload");
    double total_ms = 0;
    for (int round = 0; round < 4; ++round) {
        for (size_t i = 0; i < sizes.size(); ++i) {
            std::cout << "UPLOAD_BEGIN round=" << round << " tensor=" << i << " bytes=" << sizes[i] << '\n';
            const auto start = std::chrono::steady_clock::now();
            source->set_backend_tensor(tensors[i], "weight_" + std::to_string(i),
                assets::TensorStorageType::F32, {int64_t(sizes[i] / 4)});
            ggml_backend_synchronize(backend);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            total_ms += ms;
            std::cout << "UPLOAD_END round=" << round << " tensor=" << i << " ms=" << ms << '\n';
            rss("after_upload");
            for (size_t offset = 0; offset < sizes[i] / 4; offset += readback.size()) {
                const size_t count = std::min(readback.size(), sizes[i] / 4 - offset);
                ggml_backend_tensor_get(tensors[i], readback.data(), offset * 4, count * 4);
                for (size_t j = 0; j < count; ++j) {
                    if (readback[j] != expected(i, offset + j)) {
                        throw std::runtime_error("readback mismatch: tensor=" + std::to_string(i) + " word=" + std::to_string(offset + j));
                    }
                }
            }
            std::cout << "VERIFY_OK round=" << round << " tensor=" << i << '\n';
        }
    }
    source->release_storage();
    rss("after_mapping_release");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::cout << "PASS file_bytes=" << file_bytes << " rounds=4 upload_ms=" << total_ms << '\n';
}
}

int main(int argc, char ** argv) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try {
        if (argc != 3) throw std::runtime_error("usage: hip_upload_probe generate|hip|cpu FILE");
        const std::string mode = argv[1];
        if (mode == "generate") generate(argv[2]);
        else if (mode == "hip" || mode == "cpu") upload(argv[2], mode == "cpu");
        else throw std::runtime_error("unknown mode");
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "FAIL " << ex.what() << '\n';
        return 1;
    }
}
