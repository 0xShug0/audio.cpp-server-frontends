#!/usr/bin/env python3
"""Compare published Ming TTS packages on the same DevLab machine, sequentially."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave
import zipfile


RELEASE = 'v0.9.3'
SOURCE_SHA = '41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48'
MODEL_REVISION = '2eb8d37c7f10672e69c8aa33091baa04c461b66c'
MODEL_SHA256 = '43f108a4e6bb54af13071d0cb345a094e0b011b3b8318d8fb2c61753733f5312'
TEXTS = {
    'short': 'The morning train arrives at the quiet station. Please bring your ticket and meet us beside the blue clock.',
    'medium': 'Good evening, and welcome to our weekly science program. Today we are exploring how sound travels through the air. When a bell rings, it creates tiny changes in pressure. These waves move away from the bell until they reach our ears. The shape of a room can change what we hear, making the same voice sound warm, distant, or surprisingly clear.',
    'long': 'On Saturday morning, we left the city before sunrise and followed the river toward the mountains. The road was quiet, with only a few delivery trucks passing through the small villages. We stopped at a bakery for fresh bread and coffee, then walked to an old wooden bridge. Below us, the water moved slowly around smooth stones. A family of ducks crossed from one bank to the other. By the time we reached the trail, sunlight had filled the valley. We checked the map, packed away our jackets, and began the climb. There was no need to hurry; we had the whole day ahead of us.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    windows = sys.platform == 'win32'
    env = os.environ.copy()
    assert env.get('CONDA_DEFAULT_ENV') == 'qwen3-tts', env.get('CONDA_DEFAULT_ENV')
    env['OMP_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '4'
    provenance = dict(release=RELEASE, source_sha=SOURCE_SHA, platform=platform.platform(),
                      runner=env.get('RUNNER_NAME'), python=sys.executable, conda=env['CONDA_DEFAULT_ENV'],
                      cpu=platform.processor(), model_revision=MODEL_REVISION, model_sha256=MODEL_SHA256,
                      threads=4, seed=1895, texts=TEXTS, packages={})

    def run(command, name, cwd=root, timeout=1200):
        command = [str(x) for x in command]
        print('+', subprocess.list2cmdline(command), flush=True)
        (results / f'{name}.command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        started = time.monotonic()
        with (results / f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
        text = (results / f'{name}.log').read_text(encoding='utf-8', errors='replace')
        print(text[-3500:], flush=True)
        return text, time.monotonic() - started

    def download(url, destination, expected):
        run(['curl', '--fail', '--location', '--retry', '3', '--max-time', '900',
             '--output', destination, url], 'download-' + destination.name)
        with destination.open('rb') as file:
            actual = hashlib.file_digest(file, 'sha256').hexdigest()
        assert actual == expected, (destination, actual, expected)

    # Install only the published package's ROCm runtime; never compile audio.cpp.
    run([sys.executable, '-m', 'pip', 'install', '--index-url',
         'https://stable.repo.amd.com/rocm/whl-next/',
         'rocm[libraries,device-gfx1151]==10.0.0'], 'runtime-install')
    discover = "import json,rocm_sdk; print(json.dumps(list(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries('amdhip64','hipblas','hipblaslt')))))"
    runtime_paths = json.loads(subprocess.check_output([sys.executable, '-c', discover], text=True))
    provenance['runtime_paths'] = runtime_paths
    if windows:
        env['PATH'] = os.pathsep.join(runtime_paths) + os.pathsep + env['PATH']
    else:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(runtime_paths)
    for key in ('CMAKE_PREFIX_PATH', 'HIP_PATH', 'ROCM_PATH', 'GGML_BACKEND_PATH'):
        env.pop(key, None)

    with urllib.request.urlopen(f'https://api.github.com/repos/0xShug0/audio.cpp/releases/tags/{RELEASE}') as response:
        release = json.load(response)
    (results / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
    archives = {asset['name']: asset for asset in release['assets']}
    suffix = '.zip' if windows else '.tar.gz'
    os_tag = 'windows' if windows else 'ubuntu'
    binaries = {}
    for backend, package_backend in [('hip', 'rocm10.0.0-test'), ('vulkan', 'vulkan')]:
        name = f'audio-{RELEASE}-bin-{os_tag}-x64-{package_backend}{suffix}'
        asset = archives[name]
        assert asset['digest'].startswith('sha256:'), asset
        archive = root / name
        download(asset['browser_download_url'], archive, asset['digest'].split(':', 1)[1])
        target = root / backend
        target.mkdir()
        if windows:
            with zipfile.ZipFile(archive) as package:
                package.extractall(target)
        else:
            with tarfile.open(archive) as package:
                package.extractall(target, filter='data')
        candidates = list(target.rglob('audiocpp_cli.exe' if windows else 'audiocpp_cli'))
        assert len(candidates) == 1, candidates
        cli = candidates[0]
        version, _ = run([cli, '--version'], backend + '-version', cwd=cli.parent)
        assert '0.9.3' in version and SOURCE_SHA[:8] in version, version
        devices, _ = run([cli, '--list-devices'], backend + '-devices', cwd=cli.parent)
        assert 'Radeon' in devices or 'gfx1151' in devices, devices
        binaries[backend] = cli
        provenance['packages'][backend] = dict(name=name, sha256=asset['digest'], version=version, devices=devices)
    (results / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    model = root / 'ming-3.0-tts-0.5b-bf16.gguf'
    download(f'https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF/resolve/{MODEL_REVISION}/{model.name}',
             model, MODEL_SHA256)
    rows = []

    def generate(backend, case, repetition, warmup=False):
        name = f'{backend}-{case}-r{repetition}' + ('-warmup' if warmup else '')
        wav_path = results / (name + '.wav')
        cli = binaries[backend]
        text, process_sec = run([cli, '--family', 'ming3_tts', '--task', 'tts', '--mode', 'offline',
            '--model', model, '--backend', backend, '--device', '0', '--threads', '4',
            '--text', TEXTS[case], '--request-option', 'language=en', '--request-option', 'seed=1895',
            '--out', wav_path, '--log', '--metrics'], name, cwd=cli.parent)
        timings = re.findall(r'\bsession\.wall_ms\s+([0-9.eE+-]+)', text)
        assert timings, 'Missing canonical session.wall_ms: ' + name
        wall_ms = float(timings[-1])
        with wave.open(str(wav_path)) as wav:
            duration = wav.getnframes() / wav.getframerate()
            assert duration > 1 and any(wav.readframes(wav.getnframes())), name
        row = dict(backend=backend, case=case, repetition=repetition, warmup=warmup,
                   session_wall_ms=wall_ms, audio_sec=duration, rtf=wall_ms / 1000 / duration,
                   process_sec=process_sec, wav=wav_path.name)
        rows.append(row)
        (results / 'measurements.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)

    for backend in binaries:
        generate(backend, 'short', 0, warmup=True)
    for repetition in range(1, 3):
        for case in TEXTS:
            for backend in (['hip', 'vulkan'] if repetition == 1 else ['vulkan', 'hip']):
                generate(backend, case, repetition)
    summary = {}
    for backend in binaries:
        measured = [row for row in rows if row['backend'] == backend and not row['warmup']]
        summary[backend] = dict(rtf=sum(r['session_wall_ms'] for r in measured) / 1000 /
                               sum(r['audio_sec'] for r in measured), runs=len(measured),
                               audio_sec=sum(r['audio_sec'] for r in measured))
    summary['hip_rtf_change_pct_vs_vulkan'] = (summary['hip']['rtf'] / summary['vulkan']['rtf'] - 1) * 100
    (results / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as output:
        output.write('```json\n' + json.dumps(summary, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()
