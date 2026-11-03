#!/usr/bin/env python3
"""Measure RE-USE and SoulX using downloaded HIP release archives, sequentially."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import wave

SOURCE_SHA = 'be1286e0bca0cc8c8e6390d59ca6041570f87a85'
MODELS = {
    'reuse': ('RE-USE-GGUF', 'f3edd063e665993410f277b7c72635dae56326e6',
              'reuse-nemotronlabs-f32.gguf', '81f47a42fe9e67e123b9a2c16803193d4d28438432ef4a0a063692615f0913f3'),
    'soulx_flashhead': ('SoulX-FlashHead-GGUF', 'cc83157045008de647a4ba11ca5ed19e66caa575',
                        'SoulX-FlashHead-Lite-Q8_0.gguf', '29f186f666a4b1ee8af257a2163d521545d94c600c46d97998c57d6fefc95751'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    assert env.get('CONDA_DEFAULT_ENV') == 'qwen3-tts', env.get('CONDA_DEFAULT_ENV')
    env['OMP_NUM_THREADS'] = '8'
    env['OPENBLAS_NUM_THREADS'] = '8'

    def run(command, name, cwd=root, timeout=1800):
        command = [str(x) for x in command]
        print('+', subprocess.list2cmdline(command), flush=True)
        (results / f'{name}.command.json').write_text(json.dumps(command, indent=2))
        with (results / f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=cwd, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        text = (results / f'{name}.log').read_text(errors='replace')
        print(text[-3500:], flush=True)
        return text

    def download(url, destination, expected=None):
        run(['curl', '--fail', '--location', '--retry', '3', '--max-time', '900',
             '--output', destination, url], 'download-' + destination.name)
        with destination.open('rb') as file:
            actual = hashlib.file_digest(file, 'sha256').hexdigest()
        if expected:
            assert actual == expected, (destination, actual, expected)
        return actual

    # Same matching runtime installation and discovery used by the Ming prebuilt runner.
    run([sys.executable, '-m', 'pip', 'install', '--index-url',
         'https://stable.repo.amd.com/rocm/whl-next/',
         'rocm[libraries,device-gfx1151]==10.0.0'], 'runtime-install')
    discover = "import json,rocm_sdk; print(json.dumps(list(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries('amdhip64','hipblas','hipblaslt')))))"
    runtime_paths = json.loads(subprocess.check_output([sys.executable, '-c', discover], text=True))
    env['LD_LIBRARY_PATH'] = os.pathsep.join(runtime_paths)
    for key in ('CMAKE_PREFIX_PATH', 'HIP_PATH', 'ROCM_PATH', 'GGML_BACKEND_PATH'):
        env.pop(key, None)

    packages = root / 'packages'
    for line in (packages / 'SHA256SUMS').read_text().splitlines():
        expected, filename = line.split(None, 1)
        with (packages / filename.strip()).open('rb') as file:
            assert hashlib.file_digest(file, 'sha256').hexdigest() == expected, filename
    archive, = packages.glob('*.tar.gz')
    binary = root / 'binary'
    binary.mkdir()
    with tarfile.open(archive) as package:
        package.extractall(binary, filter='data')
    manifest = json.loads((binary / 'BUILD.json').read_text())
    assert manifest['source_sha'] == SOURCE_SHA, manifest
    assert 'gfx1151' in manifest['gpu_targets'], manifest
    (results / 'BUILD.json').write_text(json.dumps(manifest, indent=2))
    (results / 'runtime.json').write_text(json.dumps({'library_paths': runtime_paths}, indent=2))
    cli = binary / 'audiocpp_cli'
    run([cli, '--version'], 'version', cwd=binary)
    run([cli, '--list-devices'], 'devices', cwd=binary)

    inputs = root / 'inputs'
    inputs.mkdir()
    source_url = f'https://raw.githubusercontent.com/0xShug0/audio.cpp/{SOURCE_SHA}/assets/resources/'
    hashes = {}
    for filename in ['a.wav', 'vibevoice_two_speaker_longform.wav']:
        hashes[filename] = download(source_url + filename, inputs / filename)
    with wave.open(str(inputs / 'vibevoice_two_speaker_longform.wav')) as source:
        with wave.open(str(inputs / 'reuse-60s.wav'), 'wb') as dest:
            dest.setparams(source.getparams())
            dest.writeframes(source.readframes(source.getframerate() * 60))
    soulx_repo, soulx_revision, _, _ = MODELS['soulx_flashhead']
    hashes['portrait.png'] = download(
        f'https://huggingface.co/audio-cpp/{soulx_repo}/resolve/{soulx_revision}/demo/portrait.png',
        inputs / 'portrait.png')
    (results / 'inputs.json').write_text(json.dumps(hashes, indent=2))
    rows, failed = [], set()
    for family, (repo, revision, filename, sha256) in MODELS.items():
        model = inputs / filename
        download(f'https://huggingface.co/audio-cpp/{repo}/resolve/{revision}/{filename}', model, sha256)
        audio = inputs / ('reuse-60s.wav' if family == 'reuse' else 'a.wav')
        options = {} if family == 'reuse' else {
            'reference_image_file': str(inputs / 'portrait.png'), 'seed': '42',
            'num_inference_steps': '4', 'height': '512', 'width': '512',
        }
        sequence = inputs / (family + '-requests.json')
        sequence.write_text(json.dumps([
            {'id': name, 'audio': str(audio), 'options': options}
            for name in ['warmup', 'measured']], indent=2))
        for index, impl in enumerate(['default', 'gfx1151_v1', 'gfx1151_v1', 'default']):
            if (family, impl) in failed:
                print(f'Skipping duplicate failed case {family}/{impl}', flush=True)
                continue
            name = f'{family}-{index}-{impl}'
            output = results / name
            command = [cli, '--family', family, '--task', 's2s' if family == 'reuse' else 'gen',
                       '--mode', 'offline', '--model', model, '--backend', 'hip', '--device', '0',
                       '--threads', '8', '--request-sequence', sequence, '--out-dir', output, '--log']
            if impl != 'default':
                command += ['--backend-impl', impl]
            row = dict(model=family, implementation=impl, order=index, model_revision=revision,
                       model_sha256=sha256, input=audio.name)
            try:
                text = run(command, name, cwd=binary)
                if impl != 'default':
                    assert 'implementation=gfx1151_v1' in text, name
                timings = [float(x) for x in re.findall(r'\bsession\.wall_ms\s+([0-9.eE+-]+)', text)]
                assert len(timings) == 2 and all(math.isfinite(x) and x > 0 for x in timings), timings
                if family == 'reuse':
                    with wave.open(str(output / 'measured.wav')) as wav:
                        duration = wav.getnframes() / wav.getframerate()
                        samples = array.array('h', wav.readframes(wav.getnframes()))
                        assert duration == 60 and max(abs(x) for x in samples) > 0
                else:
                    frames = 0
                    for path in (output / 'measured').rglob('*.json'):
                        artifact = json.loads(path.read_text())
                        meta = artifact.get('meta', {})
                        if meta.get('format') != 'rgb24':
                            continue
                        count = int(meta['frames'])
                        size = count * int(meta['width']) * int(meta['height']) * 3
                        assert size == artifact['bytes'] and len(artifact['payload_hex']) == 2 * size, path
                        frames += count
                    assert frames > 0, 'Missing generated RGB frames'
                    duration = frames / 25
                    row['video_frames'] = frames
                row.update(status='passed', warmup_ms=timings[0], session_wall_ms=timings[1],
                           output_seconds=duration, rtf=timings[1] / 1000 / duration)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, AssertionError, OSError, ValueError) as error:
                row.update(status='failed', error=str(error))
                failed.add((family, impl))
                print(json.dumps(row), flush=True)
            finally:
                # Retain logs and measurements, not hundreds of MB of RGB hex dumps.
                if family == 'soulx_flashhead' and output.exists():
                    shutil.rmtree(output)
            rows.append(row)
            (results / 'measurements.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    summary = {}
    for family in MODELS:
        summary[family] = {}
        for impl in ['default', 'gfx1151_v1']:
            selected = [r for r in rows if r['model'] == family and r['implementation'] == impl and r['status'] == 'passed']
            if selected:
                summary[family][impl] = dict(rtf=sum(r['rtf'] for r in selected) / len(selected), runs=len(selected))
    (results / 'summary.json').write_text(json.dumps(summary, indent=2))
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
        output.write('```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    if failed:
        raise SystemExit('Some inference cases failed; see measurements.json and per-case logs')


if __name__ == '__main__':
    main()
