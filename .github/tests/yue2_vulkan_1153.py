"""Prepare and inspect the ten-song YuE2 Vulkan reproduction using the existing campaign runner."""
import argparse
import audioop
import hashlib
import json
import math
from pathlib import Path
import re
import wave


def prepare(root):
    here = Path(__file__).resolve().parent
    cases = json.loads((here / 'yue2-vulkan-1153-cases.json').read_text())
    models = json.loads((here / 'yue2-vulkan-1153-models.json').read_text())
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    requests = inputs / 'yue2-requests.json'
    requests.write_text(json.dumps(cases, indent=2) + '\n')
    (inputs / 'model-provenance.json').write_text(json.dumps(models, indent=2) + '\n')
    hashes = [r['sha256'] + '  ' + r['destination'] for r in models if r['sha256']]
    hashes.append(hashlib.sha256(requests.read_bytes()).hexdigest() + '  inputs/yue2-requests.json')
    (inputs / 'input-sha256.txt').write_text('\n'.join(hashes) + '\n')
    results = root.parent / 'results'
    for folder in ('wavs', 'timing'):
        (results / folder).mkdir(parents=True, exist_ok=True)
    commands = []
    for case in cases:
        name = f'{case["id"]:02d}-seed-{case["seed"]}'
        argv = ['build/debug/bin/audiocpp_cli', '--backend', 'vulkan', '--device', '0',
                '--threads', '8', '--family', 'yue2', '--task', 'gen', '--model', 'inputs/yue2',
                '--seed', str(case['seed']), '--session-option', 'yue2.model_gguf=yue2-3b-q4_0.gguf',
                '--session-option', 'yue2.vae_gguf=yue2-vae-f16.gguf',
                '--session-option', 'yue2.perf_mode=off', '--session-option', 'yue2.kv_cache_type=f32',
                '--session-option', 'yue2.conv_weight_type=f32',
                '--request-option', 'semantic_temperature=0.8', '--request-option', 'semantic_top_p=0.93',
                '--request-option', 'semantic_repetition_penalty=1.1', '--request-option', 'cot=off',
                '--request-option', 'style=' + case['style'], '--lyrics', '[Instrumental]',
                '--out', '../results/wavs/' + name + '.wav', '--metrics', '--log',
                '--log-file', '../results/timing/' + name + '.log']
        commands.append({'id': name, 'argv': argv})
    manifest = {'version': 1, 'builds': [{'id': 'vulkan', 'build_type': 'Debug', 'timeout_s': 1200,
        'commands': [['cmake', '-S', '.', '-B', 'build/debug', '-G', 'Ninja',
                      '-DCMAKE_BUILD_TYPE=Debug', '-DCMAKE_CXX_STANDARD=17',
                      '-DENGINE_ENABLE_CUDA=OFF', '-DENGINE_ENABLE_HIP=OFF',
                      '-DENGINE_ENABLE_VULKAN=ON', '-DENGINE_BUILD_TESTS=OFF',
                      '-DENGINE_BUILD_WARMBENCH=OFF', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF',
                      '-DAUDIOCPP_MODEL_SET=custom', '-DAUDIOCPP_MODELS=yue2',
                      '-DVulkan_LIBRARY=/lib/x86_64-linux-gnu/libvulkan.so.1'],
                     ['cmake', '--build', 'build/debug', '--target', 'audiocpp_cli', '--parallel', '8']]}],
        'suites': [{'id': 'correctness', 'build': 'vulkan', 'gpu': True, 'timeout_s': 900,
                    'cases': commands},
                   {'id': 'framework', 'build': 'vulkan', 'timeout_s': 120,
                    'cases': [{'id': 'audio-results', 'argv': ['python3', '.github/tests/yue2_vulkan_1153.py',
                              '--root', '.', '--analyze']}]}],
        'health_check': {'argv': ['vulkaninfo', '--summary'], 'timeout_s': 20}}
    (root / 'campaign.json').write_text(json.dumps(manifest, indent=2) + '\n')


def inspect_wav(path):
    with wave.open(str(path), 'rb') as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        assert channels == 2 and width == 2 and frames > 0, (channels, width, frames)
        sums = [0, 0]
        squares = [0, 0]
        peaks = [0, 0]
        seen = quiet = streak = longest = 0
        while block := wav.readframes(rate):
            count = len(block) // (width * channels)
            rms = []
            for ch in (0, 1):
                pcm = audioop.tomono(block, width, 1 if ch == 0 else 0, 1 if ch == 1 else 0)
                value = audioop.rms(pcm, width)
                rms.append(value)
                sums[ch] += audioop.avg(pcm, width) * count
                squares[ch] += value * value * count
                peaks[ch] = max(peaks[ch], audioop.max(pcm, width))
            silent = max(rms) == 0
            quiet += int(silent and count == rate)
            streak = streak + 1 if silent and count == rate else 0
            longest = max(longest, streak)
            seen += count
        assert seen == frames
    mean = [s / frames / 32768 for s in sums]
    rms = [math.sqrt(s / frames) / 32768 for s in squares]
    return {'seconds': frames / rate, 'sample_rate': rate, 'channels': channels,
            'rms': rms, 'mean': mean, 'peak': [v / 32768 for v in peaks],
            'silent_file': max(rms) < 1e-6, 'full_scale_dc': any(abs(v) >= .99 for v in mean),
            'silent_one_second_windows': quiet, 'longest_silent_run_seconds': longest}


def analyze(root):
    rows = []
    results = root.parent / 'results'
    cases = json.loads((root / 'inputs/yue2-requests.json').read_text())
    for case in cases:
        name = f'{case["id"]:02d}-seed-{case["seed"]}'
        wav = results / 'wavs' / (name + '.wav')
        row = dict(case)
        if not wav.exists():
            row['error'] = 'No output WAV; see campaign result and case log'
        else:
            row.update(inspect_wav(wav))
            log = (results / 'timing' / (name + '.log')).read_text()
            sampling = next(line for line in log.splitlines() if 'yue2.sampling.semantic' in line)
            assert 'temperature=0.8 top_p=0.93' in sampling and 'repetition_penalty=1.1' in sampling
            assert 'yue2.ar.weights.buffer_name Vulkan0' in log
            row['semantic_tokens'] = int(re.findall(r'yue2.semantic.tokens (\d+)', log)[-1])
            row['session.wall_ms'] = float(re.findall(r'session.wall_ms ([0-9.]+)', log)[-1])
            row['default_token_cap'] = row['semantic_tokens'] == 9000
        rows.append(row)
    (results / 'audio-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
    return int(any(r.get('error') or r.get('silent_file') or r.get('full_scale_dc') for r in rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare(args.root.resolve())
    elif args.analyze:
        raise SystemExit(analyze(args.root.resolve()))
    else:
        p.error('choose --prepare or --analyze')
