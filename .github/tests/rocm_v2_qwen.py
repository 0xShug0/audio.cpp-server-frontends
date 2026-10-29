"""Qwen3-TTS model campaign for the existing ROCm-v2 runner controller."""
import argparse
import array
import csv
import math
import re
import statistics
import wave
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

MODEL_REPO = 'audio-cpp/audio.cpp-gguf'
MODEL_FILE = 'Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf'


TEXTS = [
    ('warmup', 'The next train leaves in ten minutes.'),
    ('short', 'Please bring the blue folder to the meeting tomorrow morning. We will review the plans over a cup of coffee.'),
    ('medium', 'The morning train arrived just as the rain stopped. Outside the station, a woman opened her little bookshop and placed a wooden sign beside the door. She had spent the evening arranging a new collection of stories, and now she was waiting for the first curious reader to step inside.'),
    ('long', 'At the edge of the village stood a small observatory. Every evening, Mara climbed the stairs and opened the heavy shutters before the stars appeared. She wrote down what she saw in a notebook, including the weather, the brightness of the moon, and any unusual lights above the hills. One evening a group of children came to visit. They expected a complicated lesson, but she simply asked them to look through the telescope. There was Saturn, surrounded by its pale rings. For a moment nobody spoke. Then one child asked whether the planet had always been waiting there for them. Mara smiled and said that some things become new only when we finally notice them.'),
]
REQUESTS = [dict(id=name, text=text, voice_ref='reference.wav', language='English', seed=1234,
                 options=dict(x_vector_only_mode=True))
            for name, text in [*TEXTS, ('short_repeat', TEXTS[1][1])]]


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
    with model.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert model.stat().st_size == info['size'], (model.stat().st_size, info['size'])
    assert digest == info['lfs']['oid'], (digest, info)
    (inputs / 'model-provenance.json').write_text(json.dumps(dict(repo=MODEL_REPO, revision=revision, file=MODEL_FILE, sha256=digest, size=model.stat().st_size), indent=2))
    with (inputs / 'reference.wav').open('wb') as f:
        subprocess.run(['git', 'show', 'HEAD:assets/resources/a.wav'], cwd=root, stdout=f, check=True)
    (inputs / 'qwen-requests.json').write_text(json.dumps(REQUESTS, indent=2))
    (inputs / 'qwen-trace.json').write_text(json.dumps(REQUESTS[:2], indent=2))
    hashes = [digest + '  inputs/' + model.name]
    for name in ['reference.wav', 'qwen-requests.json', 'qwen-trace.json']:
        with (inputs / name).open('rb') as stream:
            hashes.append(hashlib.file_digest(stream, 'sha256').hexdigest() + '  inputs/' + name)
    (inputs / 'input-sha256.txt').write_text('\n'.join(hashes) + '\n')
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
    builds = [dict(id='device-health', build_type='Debug', timeout_s=120, commands=[['cmake', '-E', 'make_directory', 'build/debug'], ['/opt/rocm/bin/hipcc', '--offload-arch=gfx1151', '-g', '-O2', '-std=c++17', '.github/tests/hip_vmm_probe.cpp', '-o', 'build/debug/device-health']]), dict(id='model', build_type='Debug', timeout_s=1800, commands=[
        configure, ['cmake', '--build', 'build/debug', '--target', 'audiocpp_cli', '--parallel', '8']])]
    def command(backend, sequence, out):
        return ['build/debug/bin/audiocpp_cli',
            '--task', 'tts', '--family', 'qwen3_tts', '--backend', backend, '--threads', '8',
            '--model', 'inputs/qwen3-0.6b-base-q8_0.gguf', '--request-sequence', sequence,
            '--out-dir', out, '--log', '--metrics']
    paired = [dict(id=tag, argv=command(backend, 'inputs/qwen-requests.json', '../results/wavs/' + tag))
        for tag, backend in [('hip-a1','hip'), ('v2-b1','rocm_v2'), ('v2-b2','rocm_v2'), ('hip-a2','hip')]]
    traces = [dict(id=backend, argv=['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--output-format', 'csv',
        '--output-directory', '../results/profile/' + backend, '--'] + command(backend, 'inputs/qwen-trace.json', '../results/trace-wavs/' + backend)) for backend in ['hip','rocm_v2']]
    return dict(version=1, builds=builds, health_check=dict(argv=['build/debug/device-health', 'health'], timeout_s=15), suites=[
        dict(id='correctness', build='model', gpu=True, timeout_s=900, cases=paired),
        dict(id='framework', build='model', gpu=True, required=False, timeout_s=300, cases=traces),
        dict(id='report', build='model', gpu=False, timeout_s=120, cases=[dict(id='summary',
            argv=['python3', '.github/tests/rocm_v2_qwen.py', '--summarize', '../results'])])])


def summarize(root, implementation="rocm_v2"):
    variants = ['hip-a1', 'v2-b1', 'v2-b2', 'hip-a2']
    names = [item['id'] for item in REQUESTS]
    rows, audio = [], {}
    for variant in variants:
        text = (root / 'case/correctness' / (variant + '.log')).read_text()
        selected = ('implementation=' + implementation) in text
        assert selected == variant.startswith('v2'), (variant, 'incorrect backend selection')
        metrics, stages, pending = {}, {}, {}
        for line in text.splitlines():
            timing = re.match(r'^\[TIMING[^\]]*\] (qwen3_tts\.[\w.]+) ([\d.eE+-]+)$', line)
            if timing:
                pending[timing[1]] = pending.get(timing[1], 0.0) + float(timing[2])
            metric = re.match(r'^metrics\[([^\]]+)\]\.([^=]+)=(.*)$', line)
            if metric:
                name, key, value = metric.groups()
                if name not in stages:
                    stages[name], pending = pending, {}
                metrics.setdefault(name, {})[key] = value
        assert set(metrics) == set(names), (variant, 'missing request metrics', list(metrics))
        for name in names:
            with wave.open(str(root / 'wavs' / variant / (name + '.wav')), 'rb') as wav:
                assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
                assert wav.getframerate() == 24000
                pcm = array.array('h', wav.readframes(wav.getnframes()))
            assert pcm and any(pcm), (variant, name, 'empty or silent WAV')
            audio[variant, name] = pcm
            seconds = len(pcm) / 24000
            wall = float(metrics[name]['wall_ms'])
            assert math.isfinite(wall) and wall > 0
            assert abs(float(metrics[name]['audio_duration_ms']) - seconds * 1000) <= 1
            if name != 'warmup':
                rows.append(dict(variant=variant, backend='hip' if variant.startswith('hip') else implementation,
                    request=name, wall_ms=wall, audio_seconds=seconds, rtf=wall / (1000 * seconds),
                    talker_ms=stages[name]['qwen3_tts.talker_ms'],
                    decoder_ms=stages[name]['qwen3_tts.speech_decoder_ms']))
    per_request, aggregate = {}, {}
    for name in names[1:]:
        per_request[name] = {}
        for backend in ['hip', implementation]:
            group = [row for row in rows if row['request'] == name and row['backend'] == backend]
            assert len(group) == 2
            per_request[name][backend] = {key: statistics.mean(row[key] for row in group)
                for key in ['wall_ms', 'audio_seconds', 'rtf', 'talker_ms', 'decoder_ms']}
    for backend in ['hip', implementation]:
        group = [row for row in rows if row['backend'] == backend]
        aggregate[backend] = dict(wall_ms=sum(row['wall_ms'] for row in group),
            audio_seconds=sum(row['audio_seconds'] for row in group))
        aggregate[backend]['rtf'] = aggregate[backend]['wall_ms'] / (1000 * aggregate[backend]['audio_seconds'])
    parity = []
    for name in names[1:]:
        reference = audio['hip-a1', name]
        for variant in variants[1:]:
            candidate = audio[variant, name]
            result = dict(request=name, variant=variant, equal_length=len(reference) == len(candidate),
                          byte_identical=reference == candidate)
            if result['equal_length']:
                aa, bb = sum(x*x for x in reference), sum(x*x for x in candidate)
                error = sum((x-y)**2 for x,y in zip(reference, candidate))
                result['cosine'] = sum(x*y for x,y in zip(reference, candidate)) / math.sqrt(aa*bb)
                result['snr_db'] = 10 * math.log10(aa/error) if error else None
            parity.append(result)
    repeat = {variant: audio[variant, 'short'] == audio[variant, 'short_repeat'] for variant in variants}
    kernels = set()
    for path in (root / 'profile' / implementation).rglob('*kernel_trace.csv'):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                name = row['Kernel_Name']
                if any(key in name for key in ['rocm_v2_sgemm', 'rocm_v2_im2col', 'hip_halo_sgemm', 'hip_halo_im2col']):
                    kernels.add(name)
    result = dict(rows=rows, per_request=per_request, aggregate=aggregate, parity=parity,
        repeat_byte_identical=repeat, rocm_v2_kernel_names=sorted(kernels))
    (root / 'qwen-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(per_request=per_request, aggregate=aggregate,
        repeat_byte_identical=repeat, rocm_v2_kernel_names=sorted(kernels)), indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', type=Path)
    actions.add_argument('--summarize', type=Path)
    parser.add_argument('--implementation', default='rocm_v2')
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
    else:
        summarize(args.summarize, args.implementation)
