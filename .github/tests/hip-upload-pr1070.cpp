#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

int main(int argc, char ** argv) {
    if ((argc != 3 && argc != 4) || std::string(argv[1]) != "--log") return 2;
    engine::debug::configure_logging({true, std::nullopt});
    ggml_time_init();
    const auto started = ggml_time_us();
    const std::string mode = argc == 4 ? argv[3] : "";
    if (!mode.empty() && mode != "--prepare" && mode != "--pressure") return 2;
    const bool pressure = !mode.empty();
    std::vector<size_t> sizes = {16, 64, 65, 128, 256};
    if (pressure) sizes.insert(sizes.end(), 64, 256);
    else sizes.insert(sizes.end(), 16, 128);
    constexpr size_t MiB = 1024 * 1024;
    std::vector<unsigned char> pattern(MiB);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = (i * 17 + i / 256) % 256;
    const std::string file = argv[2];
    if (mode == "--prepare" || !std::ifstream(file).good()) {
        uint64_t payload_bytes = 0;
        for (const auto size : sizes) payload_bytes += size * MiB;
        if (std::filesystem::space(std::filesystem::absolute(file).parent_path()).available < payload_bytes + 1024 * MiB)
            throw std::runtime_error("Insufficient free disk space for upload fixture");
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
#ifndef _WIN32
    // Charge fresh file faults to the limited test cgroup, not the fixture writer.
    const int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open fixture to evict clean page cache");
    const int advice = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    if (advice != 0) throw std::runtime_error("Cannot request clean page-cache eviction");
#endif
    if (mode == "--prepare") {
        std::cout << "fixture.prepared=true tensor_count=" << sizes.size() << std::endl;
        return 0;
    }
    if (pressure) {
        constexpr uint64_t limit = uint64_t{12} * 1024 * 1024 * 1024;
#ifdef _WIN32
        if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), 1024 * 1024, static_cast<SIZE_T>(limit),
                                       QUOTA_LIMITS_HARDWS_MAX_ENABLE))
            throw std::runtime_error("SetProcessWorkingSetSizeEx failed: " + std::to_string(GetLastError()));
        SIZE_T minimum = 0, maximum = 0;
        DWORD flags = 0;
        if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &minimum, &maximum, &flags) ||
            maximum != limit || !(flags & QUOTA_LIMITS_HARDWS_MAX_ENABLE))
            throw std::runtime_error("Windows hard working-set limit was not applied");
        std::cout << "pressure.limit_kind=windows_hard_working_set pressure.limit_bytes=" << maximum << std::endl;
#else
        std::ifstream cgroup("/proc/self/cgroup");
        std::string group, line;
        while (std::getline(cgroup, line)) {
            if (line.rfind("0::", 0) == 0) group = line.substr(3);
        }
        if (group.empty()) throw std::runtime_error("Cannot identify test cgroup");
        const std::string root = "/sys/fs/cgroup" + group;
        std::ifstream memory_max(root + "/memory.max");
        uint64_t actual_limit = 0;
        memory_max >> actual_limit;
        if (!memory_max || actual_limit != limit) throw std::runtime_error("Linux 12 GiB cgroup limit was not applied");
        std::cout << "pressure.limit_kind=linux_cgroup pressure.limit_bytes=" << actual_limit
                  << " cgroup=" << group << std::endl;
#endif
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
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX counters{};
        if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters), sizeof(counters)))
            throw std::runtime_error("GetProcessMemoryInfo failed");
        std::cout << "memory.working_set_bytes=" << counters.WorkingSetSize
                  << " memory.peak_working_set_bytes=" << counters.PeakWorkingSetSize
                  << " memory.private_bytes=" << counters.PrivateUsage << std::endl;
#else
        std::ifstream status("/proc/self/status");
        for (std::string line; std::getline(status, line);) {
            if (line.rfind("RssFile:", 0) == 0 || line.rfind("RssAnon:", 0) == 0 || line.rfind("VmRSS:", 0) == 0)
                std::cout << line << std::endl;
        }
#endif
    }
    for (auto * buffer : buffers) ggml_backend_buffer_free(buffer);
    for (auto * ctx : contexts) ggml_free(ctx);
    engine::debug::timing_log_scalar("session.wall_ms", (ggml_time_us() - started) / 1000.0);
}
