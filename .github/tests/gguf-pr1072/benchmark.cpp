#include "gguf.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3 || std::string(argv[1]) != "--log") return 2;
    const auto start = std::chrono::steady_clock::now();
    if (std::string(argv[2]) == "--create") {
        if (argc != 4) return 2;
        auto * ctx = gguf_init_empty();
        std::vector<uint8_t> payload(16 * 1024 * 1024);
        for (size_t i = 0; i < payload.size(); ++i) payload[i] = uint8_t(i * 17);
        gguf_set_arr_data(ctx, "audiocpp.embedded_files.data", GGUF_TYPE_UINT8, payload.data(), payload.size());
        gguf_set_val_str(ctx, "after_array", "must survive");
        const bool ok = gguf_write_to_file(ctx, argv[3], true);
        gguf_free(ctx);
        if (!ok) return 1;
        printf("fixture.bytes=%zu\n", payload.size());
    } else {
        if (argc != 4) return 2;
        auto * ctx = gguf_init_from_file(argv[2], {true, nullptr});
        if (!ctx) return 1;
        const auto loaded = std::chrono::steady_clock::now();
        printf("gguf.read_ms=%.6f\n", std::chrono::duration<double, std::milli>(loaded - start).count());
        printf("gguf.keys=%lld\ngguf.tensors=%lld\n", (long long)gguf_get_n_kv(ctx), (long long)gguf_get_n_tensors(ctx));
        std::vector<char> meta(gguf_get_meta_size(ctx));
        gguf_get_meta_data(ctx, meta.data());
        FILE * output = fopen(argv[3], "wb");
        if (!output) { gguf_free(ctx); return 1; }
        const bool ok = fwrite(meta.data(), 1, meta.size(), output) == meta.size();
        const int closed = fclose(output);
        gguf_free(ctx);
        if (!ok || closed != 0) return 1;
    }
    printf("session.wall_ms=%.6f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return 0;
}
