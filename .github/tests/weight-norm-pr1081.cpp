#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace engine;

class CountingSource final : public assets::TensorSource {
public:
    std::shared_ptr<const assets::TensorSource> inner;
    mutable int uploads = 0;
    explicit CountingSource(std::shared_ptr<const assets::TensorSource> source) : inner(std::move(source)) {}
    const std::filesystem::path & source_path() const noexcept override { return inner->source_path(); }
    bool has_tensor(std::string_view n) const noexcept override { return inner->has_tensor(n); }
    assets::TensorMetadata require_metadata(std::string_view n) const override { return inner->require_metadata(n); }
    std::vector<assets::TensorMetadata> tensors() const override { return inner->tensors(); }
    assets::RawTensorData require_tensor_data(std::string_view n) const override { return inner->require_tensor_data(n); }
    std::vector<float> require_f32(std::string_view n, const std::optional<std::vector<int64_t>> & s) const override {
        return inner->require_f32(n, s);
    }
    std::optional<std::vector<float>> optional_f32(std::string_view n, const std::optional<std::vector<int64_t>> & s) const override {
        return inner->optional_f32(n, s);
    }
    int64_t require_i64_scalar(std::string_view n) const override { return inner->require_i64_scalar(n); }
    void set_backend_tensor(ggml_tensor * t, std::string_view n, assets::TensorStorageType type,
                            const std::vector<int64_t> & s) const override {
        ++uploads;
        inner->set_backend_tensor(t, n, type, s);
    }
};

int main(int argc, char ** argv) {
    if (argc != 4 || std::string(argv[1]) != "--log") return 2;
    debug::configure_logging({true, std::nullopt});
    ggml_time_init();
    const auto started = ggml_time_us();
    try {
        const std::string backend_name = argv[2];
        core::BackendConfig config;
        if (backend_name == "cuda") config.type = core::BackendType::Cuda;
        else if (backend_name == "vulkan") config.type = core::BackendType::Vulkan;
        else if (backend_name == "hip") config.type = core::BackendType::Hip;
        else if (backend_name == "cpu") config.type = core::BackendType::Cpu;
        else throw std::runtime_error("unknown backend");
        config.threads = 8;
        core::ExecutionContext execution(config);
        if (execution.backend_type() != config.type) throw std::runtime_error("backend mismatch");
        const std::filesystem::path root = argv[3];
        std::filesystem::create_directories(root);
        const auto safe = root / "fixture.safetensors";
        std::vector<float> plain(2048 * 4096), v(512 * 2048), g(512);
        for (size_t i = 0; i < plain.size(); ++i) plain[i] = (static_cast<int>(i % 257) - 128) / 137.0F;
        for (size_t i = 0; i < v.size(); ++i) v[i] = (static_cast<int>(i % 113) - 56) / 79.0F;
        for (size_t i = 0; i < g.size(); ++i) g[i] = 1.0F + (i % 17) / 31.0F;
        const auto bytes = [](const std::vector<float> & x) {
            std::vector<unsigned char> out(x.size() * sizeof(float));
            std::memcpy(out.data(), x.data(), out.size());
            return out;
        };
        io::write_safetensors_file(safe, {
            {"plain.weight", "F32", {2048, 4096}, bytes(plain)},
            {"norm.weight_v", "F32", {512, 2048}, bytes(v)},
            {"norm.weight_g", "F32", {512}, bytes(g)},
        });
        const auto gguf = root / "fixture.gguf";
        assets::convert_tensor_source_to_gguf(safe, gguf, assets::TensorStorageType::Q8_0, true, false);
        for (const auto & path : {safe, gguf}) {
            auto counted = std::make_shared<CountingSource>(assets::open_tensor_source(path));
            auto source = assets::make_weight_norm_folded_tensor_source(counted, {"norm.weight"});
            for (auto type : {assets::TensorStorageType::F32, assets::TensorStorageType::F16,
                              assets::TensorStorageType::BF16, assets::TensorStorageType::Q8_0,
                              assets::TensorStorageType::Native}) {
                for (const std::string name : {"plain.weight", "norm.weight"}) {
                    const auto dims = source->require_metadata(name).shape;
                    const auto expected = source->require_tensor(name, type, dims);
                    core::BackendWeightStore store(execution.backend(), config.type, "test.weight_norm", 1024 * 1024);
                    auto tensor = store.load_tensor(*source, name, type, dims);
                    store.upload();
                    ggml_backend_synchronize(execution.backend());
                    std::vector<std::byte> actual(ggml_nbytes(tensor.tensor));
                    ggml_backend_tensor_get(tensor.tensor, actual.data(), 0, actual.size());
                    // BackendWeightStore deliberately converts native BF16 to F16 on Vulkan.
                    auto reference = expected;
                    if (tensor.tensor->type != reference.type)
                        reference = source->require_tensor(name, assets::TensorStorageType::F16, dims);
                    if (actual != reference.bytes) throw std::runtime_error("uploaded bytes differ: " + path.string() + " " + name);
                    std::cout << "CHECK source=" << path.extension() << " name=" << name
                              << " request_type=" << static_cast<int>(type)
                              << " stored_type=" << ggml_type_name(tensor.tensor->type) << " exact_bytes=true\n";
                }
            }
            // The F32 convenience API is separate from load_tensor().
            {
                core::BackendWeightStore store(execution.backend(), config.type, "test.weight_norm.f32", 1024 * 1024);
                auto tensor = store.load_f32_tensor(*source, "norm.weight", {512, 2048});
                store.upload();
                std::vector<float> actual(v.size());
                ggml_backend_tensor_get(tensor.tensor, actual.data(), 0, actual.size() * sizeof(float));
                if (actual != source->require_f32("norm.weight")) throw std::runtime_error("F32 convenience differs");
                std::cout << "CHECK f32_convenience exact_bytes=true\n";
            }
            for (const std::string name : {"plain.weight", "norm.weight"}) {
                bool rejected = false;
                try {
                    core::BackendWeightStore store(execution.backend(), config.type, "test.weight_norm.bad", 1024 * 1024);
                    store.load_tensor(*source, name, assets::TensorStorageType::F32, {1024, 1024});
                    store.upload();
                } catch (const std::runtime_error &) { rejected = true; }
                if (!rejected) throw std::runtime_error("mismatched shape accepted");
                std::cout << "CHECK shape_rejected=true name=" << name << '\n';
            }
            counted->uploads = 0;
            std::vector<double> timings;
            for (int i = 0; i < 6; ++i) {
                const auto begin = ggml_time_us();
                {
                    core::BackendWeightStore store(execution.backend(), config.type, "test.weight_norm.bench", 1024 * 1024);
                    store.load_tensor(*source, "plain.weight", assets::TensorStorageType::F32, {2048, 4096});
                    store.load_tensor(*source, "norm.weight", assets::TensorStorageType::F32, {512, 2048});
                    store.upload();
                    ggml_backend_synchronize(execution.backend());
                }
                const double elapsed = (ggml_time_us() - begin) / 1000.0;
                if (i > 0) timings.push_back(elapsed);
            }
            std::sort(timings.begin(), timings.end());
            std::cout << "RESULT source=" << path.extension() << " backend=" << backend_name
                      << " median_ms=" << timings[timings.size()/2] << " direct_uploads=" << counted->uploads << '\n';
        }
        debug::timing_log_scalar("session.wall_ms", (ggml_time_us() - started) / 1000.0);
    } catch (const std::exception & e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}
