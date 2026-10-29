"""Qwen3-TTS model campaign for the existing ROCm-v2 runner controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

MODEL_REPO = 'audio-cpp/audio.cpp-gguf'
MODEL_FILE = 'Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf'


def prepare(root):
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    with urllib.request.urlopen('https://huggingface.co/api/models/' + MODEL_REPO, timeout=60) as r:
        revision = json.load(r)['sha']
    url = f'https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{MODEL_FILE}?download=true'
    model = inputs / 'qwen3-0.6b-base-q8_0.gguf'
    subprocess.run(['curl', '--fail', '--location', '--connect-timeout', '20', '--max-time', '600', '--retry', '2', '-o', str(model), url], check=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_REPO}/tree/{revision}/Qwen3-TTS-12Hz-0.6B-Base-GGUF', timeout=60) as r:
        info = next(x for x in json.load(r) if x['path'] == MODEL_FILE)
    digest = hashlib.file_digest(model.open('rb'), 'sha256').hexdigest()
    assert digest == info['lfs']['oid'], (digest, info)
    (inputs / 'model-provenance.json').write_text(json.dumps(dict(repo=MODEL_REPO, revision=revision, file=MODEL_FILE, sha256=digest, size=model.stat().st_size), indent=2))
    with (inputs / 'reference.wav').open('wb') as f:
        subprocess.run(['git', 'show', 'HEAD:assets/resources/a.wav'], cwd=root, stdout=f, check=True)
    texts = [
        ('warmup', 'The next train leaves in ten minutes.'),
        ('short', 'Please bring the blue folder to the meeting tomorrow morning. We will review the plans over a cup of coffee.'),
        ('medium', 'The morning train arrived just as the rain stopped. Outside the station, a woman opened her little bookshop and placed a wooden sign beside the door. She had spent the evening arranging a new collection of stories, and now she was waiting for the first curious reader to step inside.'),
        ('long', 'At the edge of the village stood a small observatory. Every evening, Mara climbed the stairs and opened the heavy shutters before the stars appeared. She wrote down what she saw in a notebook, including the weather, the brightness of the moon, and any unusual lights above the hills. One evening a group of children came to visit. They expected a complicated lesson, but she simply asked them to look through the telescope. There was Saturn, surrounded by its pale rings. For a moment nobody spoke. Then one child asked whether the planet had always been waiting there for them. Mara smiled and said that some things become new only when we finally notice them.'),
    ]
    rows = [dict(id=name, text=text, voice_ref='reference.wav', language='English', seed=1234,
                 options=dict(temperature=0.0, x_vector_only_mode=True)) for name, text in texts]
    rows.append({**rows[1], 'id': 'short_repeat'})
    (inputs / 'qwen-requests.json').write_text(json.dumps(rows, indent=2))
    (inputs / 'qwen-trace.json').write_text(json.dumps(rows[:2], indent=2))
    print('Pinned model:', revision, model.stat().st_size, digest, flush=True)


def manifest(root):
    configure = ['cmake', '-S', '.', '-B', 'build/debug', '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Debug', '-DCMAKE_CXX_STANDARD=17',
        '-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang', '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
        '-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++', '-DCMAKE_PREFIX_PATH=/opt/rocm',
        '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
        '-DCMAKE_HIP_FLAGS_DEBUG=-g1 -O2', '-DCMAKE_CXX_FLAGS_DEBUG=-g1 -O2', '-DCMAKE_C_FLAGS_DEBUG=-g1 -O2',
        '-DGPU_TARGETS=gfx1151', '-DENGINE_ENABLE_HIP=ON', '-DENGINE_ENABLE_CUDA=OFF', '-DENGINE_ENABLE_VULKAN=OFF',
        '-DGGML_HIP_NO_VMM=ON', '-DENGINE_ENABLE_OPENMP=ON', '-DENGINE_BUILD_EXAMPLES=OFF',
        '-DAUDIOCPP_MODEL_SET=custom', '-DAUDIOCPP_MODELS=qwen3_tts', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF',
        '-DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF', '-DENGINE_BUILD_TESTS=OFF']
    conda = 'toolchain/conda/bin/conda'
    builds = [dict(id='device-health', build_type='Debug', timeout_s=120, commands=[['cmake', '-E', 'make_directory', 'build/debug'], ['/opt/rocm/bin/hipcc', '--offload-arch=gfx1151', '-g', '-O2', '-std=c++17', '.github/tests/hip_vmm_probe.cpp', '-o', 'build/debug/device-health']]), dict(id='model', build_type='Debug', timeout_s=1800, commands=[
        ['bash', '-c', 'set -euo pipefail; bash toolchain/miniforge.sh -b -p "$PWD/toolchain/conda"; toolchain/conda/bin/conda create --offline --name qwen3-tts --clone base --yes'],
        configure, ['cmake', '--build', 'build/debug', '--target', 'audiocpp_cli', '--parallel', '8']])]
    def command(backend, sequence, out):
        return [conda, 'run', '--no-capture-output', '-n', 'qwen3-tts', 'build/debug/bin/audiocpp_cli',
            '--task', 'tts', '--family', 'qwen3_tts', '--backend', backend, '--threads', '8',
            '--model', 'inputs/qwen3-0.6b-base-q8_0.gguf', '--request-sequence', sequence,
            '--out-dir', out, '--log', '--metrics']
    paired = [dict(id=tag, argv=command(backend, 'inputs/qwen-requests.json', '../results/wavs/' + tag))
        for tag, backend in [('hip-a1','hip'), ('v2-b1','rocm_v2'), ('v2-b2','rocm_v2'), ('hip-a2','hip')]]
    traces = [dict(id=backend, argv=['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--output-format', 'csv',
        '--output-directory', '../results/profile/' + backend, '--'] + command(backend, 'inputs/qwen-trace.json', '../results/trace-wavs/' + backend)) for backend in ['hip','rocm_v2']]
    return dict(version=1, builds=builds, health_check=dict(argv=['build/debug/device-health', 'health'], timeout_s=15), suites=[
        dict(id='correctness', build='model', gpu=True, timeout_s=900, cases=paired),
        dict(id='framework', build='model', gpu=True, required=False, timeout_s=300, cases=traces)])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', type=Path, required=True)
    prepare(p.parse_args().prepare)
