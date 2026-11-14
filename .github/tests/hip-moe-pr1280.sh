#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
root="$PWD"
rocm="${ROCM_PATH:-/opt/rocm}"
export PATH="$rocm/bin:$PATH"
export LD_LIBRARY_PATH="$rocm/lib:${LD_LIBRARY_PATH:-}"
if [[ "$1" == build ]]; then
    { uname -a; lscpu; "$rocm/bin/hipcc" --version; "$rocm/bin/rocminfo"; } >results/environment.log 2>&1
    for arm in base pr; do
        rev=40c8c8686998f6a3338cf50431adda205ddcf7e9
        [[ "$arm" != pr ]] || rev=cfa718db0c9df6069cf82d9e125cc1cb49fff75f
        mkdir -p "source-$arm"
        curl -fLsS --retry 2 "https://codeload.github.com/0xShug0/audio.cpp/tar.gz/$rev" |
            tar -xz --strip-components=1 -C "source-$arm"
        printf '%s\n' "$rev" >"results/$arm-revision.txt"
        cmake -S "source-$arm" -B "build-$arm" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$rocm/llvm/bin/clang" \
            -DCMAKE_CXX_COMPILER="$rocm/llvm/bin/clang++" -DCMAKE_HIP_COMPILER="$rocm/llvm/bin/clang++" \
            -DCMAKE_PREFIX_PATH="$rocm" -DGPU_TARGETS=gfx1151 -DENGINE_ENABLE_HIP=ON \
            -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=OFF \
            -DENGINE_ENABLE_OPENMP=ON -DGGML_OPENMP=ON -DENGINE_BUILD_EXAMPLES=ON \
            -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF -DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF \
            -DAUDIOCPP_MODEL_SET=custom '-DAUDIOCPP_MODELS=fish_drama3;ming3_audio;nemotron_transcribe' \
            2>&1 | tee "results/$arm-configure.log"
        cmake --build "build-$arm" --target audiocpp_cli -j 8 2>&1 | tee "results/$arm-build.log"
    done
    exit
fi

fetch() {
    curl -fLsS --retry 2 --connect-timeout 20 --max-time 1800 \
        "https://huggingface.co/$repo/resolve/$revision/$1?download=true" -o "$weights/$1"
}

for family in fish_drama3 ming3_audio nemotron_transcribe; do
    weights="$(mktemp -d "$root/weights.XXXXXX")"
    out="$root/results/$family"
    mkdir -p "$out"
    revision=main
    task=asr
    case "$family" in
        fish_drama3)
            repo=pentacoxian-dev/drama3-small-GGUF
            file=drama3-small-IQ3_XXS.gguf
            task=tts
            ;;
        ming3_audio)
            repo=audio-cpp/Ming-3.0-Audio-Tiny-GGUF
            file=ming-3.0-audio-tiny-q4_k.gguf
            ;;
        nemotron_transcribe)
            repo=Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF
            revision=304032f14e1e9e4042f5491d2d0194c044710bb7
            file=nemotron-transcribe-30b-a4b-experts-Q2_K.gguf
            ;;
    esac
    printf '%s %s %s\n' "$repo" "$revision" "$file" >"$out/package.txt"
    if ! fetch "$file" >"$out/download.log" 2>&1 ||
       { [[ "$family" == nemotron_transcribe ]] && ! fetch mmproj-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf >>"$out/download.log" 2>&1; }; then
        echo download_failed >"$out/status.txt"
        rm -rf "$weights"
        continue
    fi
    python3 - "$root" "$task" >"$out/requests.json" <<'PY'
import json, sys
from pathlib import Path
root, task = sys.argv[1:]
if task == "tts":
    a = {"text": "The lighthouse shines across the bay. Tonight the sea is calm and clear. We will sail home before sunrise, when the harbor lights are still glowing and the streets are quiet.", "seed": 42}
    b = {"text": "We waited on the platform until the last train arrived. By then the rain had stopped, and the evening sky was bright above the station.", "seed": 42}
else:
    audio = Path(root) / "source-base/assets/asr_validation/librispeech"
    a = {"audio": str(audio / "librispeech_test_clean_6930-75918-0001.wav")}
    b = {"audio": str(audio / "librispeech_test_other_7902-96591-0001.wav")}
print(json.dumps([dict(item, id=name) for name, item in [("warmup", a), ("repeat", a), ("different", b)]]))
PY
    cases=(default disk hybrid)
    [[ "$family" != fish_drama3 ]] || cases+=(cpu_layers)
    for mode in "${cases[@]}"; do
        for arm in base pr; do
            prefix="$family."
            [[ "$arm" != pr ]] || prefix=moe_
            opts=()
            if [[ "$mode" == disk || "$mode" == hybrid ]]; then
                opts+=(--session-option "${prefix}disk_experts=true")
                [[ "$family" != fish_drama3 ]] || opts+=(--session-option "${prefix}cpu_expert_layers=0")
            fi
            [[ "$mode" != hybrid ]] || opts+=(--session-option moe_cpu_share=0.5)
            [[ "$mode" != cpu_layers ]] || opts+=(--session-option "${prefix}cpu_expert_layers=1")
            dir="$out/$mode-$arm"
            mkdir -p "$dir"
            rc=0
            timeout 600 "$root/build-$arm/bin/audiocpp_cli" --task "$task" --family "$family" \
                --backend hip --threads 8 --model "$weights/$file" --request-sequence "$out/requests.json" \
                --out-dir "$dir" --metrics --log "${opts[@]}" >"$dir/run.log" 2>&1 || rc=$?
            printf '%s\n' "$rc" >"$dir/exit-code.txt"
        done
    done
    if [[ "$family" == fish_drama3 ]]; then
        rc=0
        timeout 120 "$root/build-pr/bin/audiocpp_cli" --task tts --family "$family" \
            --backend hip --model "$weights/$file" --text "This request should be rejected." \
            --session-option moe_disk_experts=true --session-option fish_drama3.disk_experts=true \
            --log >"$out/conflict.log" 2>&1 || rc=$?
        printf '%s\n' "$rc" >"$out/conflict-exit-code.txt"
    fi
    rm -rf "$weights"
done
