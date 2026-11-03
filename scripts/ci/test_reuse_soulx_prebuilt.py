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


def parse_measurement(text, family, output, verify_files=True):
    # Match the top-level key, not ltx_video_vae.session.wall_ms.
    timings = [float(x) for x in re.findall(
        r'(?m)^\[TIMING(?: [^\]]+)?\]\s+session\.wall_ms\s+([0-9.eE+-]+)[ \t]*$', text)]
    assert len(timings) == 2 and all(math.isfinite(x) and x > 0 for x in timings), timings
    measured = text.split('\nrequest_id=measured\n')
    assert len(measured) == 2, 'Missing measured request output'
    if family == 'reuse':
        with wave.open(str(output / 'measured.wav')) as wav:
            duration = wav.getnframes() / wav.getframerate()
            samples = array.array('h', wav.readframes(wav.getnframes()))
            assert duration == 60 and max(abs(x) for x in samples) > 0
        extra = {}
    else:
        sizes = [int(x) for x in re.findall(
            r'(?m)^artifact=soulx_flashhead_video_\d+ kind=custom bytes=(\d+)$', measured[1])]
        frame_bytes = 512 * 512 * 3
        assert sizes and all(size > 0 and size % frame_bytes == 0 for size in sizes), sizes
        frames = sum(sizes) // frame_bytes
        if verify_files:
            files = list((output / 'measured').glob('*.json'))
            assert len(files) == len(sizes), 'Missing video artifacts'
            saved_frames = 0
            for path in files:
                artifact = json.loads(path.read_text())
                meta = artifact['meta']
                assert meta['format'] == 'rgb24' and meta['fps'] == '25', meta
                assert int(meta['width']) == 512 and int(meta['height']) == 512, meta
                count = int(meta['frames'])
                assert artifact['bytes'] == count * frame_bytes and len(artifact['payload_hex']) == 2 * artifact['bytes'], path
                saved_frames += count
            assert saved_frames == frames
        duration = frames / 25
        extra = {'video_frames': frames}
    return dict(status='passed', warmup_ms=timings[0], session_wall_ms=timings[1],
                output_seconds=duration, rtf=timings[1] / 1000 / duration, **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    if (results / 'measurements.json').exists():
        previous = json.loads((results / 'BUILD.json').read_text())
        assert previous['source_sha'] == SOURCE_SHA, previous
        rows = json.loads((results / 'measurements.json').read_text())
        for row in rows:
            if 'order' not in row:
                continue
            family, impl, index = row['model'], row['implementation'], row['order']
            assert row['model_sha256'] == MODELS[family][3], row
            try:
                text = (results / f'{family}-{index}-{impl}.log').read_text()
                if family == 'reuse' and 'RE-USE is disabled on HIP: this backend has not been validated' in text:
                    row.update(status='blocked', error='Prebuilt explicitly disables RE-USE on HIP')
                elif family == 'soulx_flashhead' and row['status'] == 'failed' and row['error'].startswith('['):
                    # Inference completed; only the original timing assertion failed.
                    if impl != 'default':
                        assert 'implementation=gfx1151_v1' in text
                    row.update(parse_measurement(text, family, results / f'{family}-{index}-{impl}', False))
                    row.pop('error')
                    row['recovered_from_log'] = True
            except Exception as error:
                row.update(status='failed', error=f'Previous result recovery failed: {error}')
            print('Retained previous case:', json.dumps(row), flush=True)
        (results / 'measurements.json').write_text(json.dumps(rows, indent=2))
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
    for family, (repo, revision, filename, sha256) in MODELS.items():
        if any(r['model'] == family and r['status'] == 'blocked' for r in rows):
            print(f'{family}: retained explicit unsupported-backend result; no repeated inference', flush=True)
            continue
        pending = [(i, impl) for i, impl in enumerate(['default', 'gfx1151_v1', 'gfx1151_v1', 'default'])
                   if not any(r['model'] == family and r.get('order') == i and r['status'] == 'passed' for r in rows)]
        if not pending:
            continue
        model = inputs / filename
        try:
            download(f'https://huggingface.co/audio-cpp/{repo}/resolve/{revision}/{filename}', model, sha256)
            audio_name = 'vibevoice_two_speaker_longform.wav' if family == 'reuse' else 'a.wav'
            download(source_url + audio_name, inputs / audio_name)
            if family == 'reuse':
                with wave.open(str(inputs / audio_name)) as source:
                    with wave.open(str(inputs / 'reuse-60s.wav'), 'wb') as dest:
                        dest.setparams(source.getparams())
                        dest.writeframes(source.readframes(source.getframerate() * 60))
            else:
                download(f'https://huggingface.co/audio-cpp/{repo}/resolve/{revision}/demo/portrait.png',
                         inputs / 'portrait.png')
            rows = [r for r in rows if not (r['model'] == family and r.get('phase') == 'download')]
        except Exception as error:
            rows.append(dict(model=family, phase='download', status='failed', error=str(error)))
            (results / 'measurements.json').write_text(json.dumps(rows, indent=2))
            print(f'{family} setup failed: {error}', flush=True)
            continue
        audio = inputs / ('reuse-60s.wav' if family == 'reuse' else 'a.wav')
        options = {} if family == 'reuse' else {
            'reference_image_file': str(inputs / 'portrait.png'), 'seed': '42',
            'num_inference_steps': '4', 'height': '512', 'width': '512',
        }
        sequence = inputs / (family + '-requests.json')
        sequence.write_text(json.dumps([
            {'id': name, 'audio': str(audio), 'options': options}
            for name in ['warmup', 'measured']], indent=2))
        for index, impl in pending:
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
                row.update(parse_measurement(text, family, output))
            except Exception as error:
                row.update(status='failed', error=str(error))
                print(json.dumps(row), flush=True)
            finally:
                # Retain logs and measurements, not hundreds of MB of RGB hex dumps.
                if family == 'soulx_flashhead' and output.exists():
                    shutil.rmtree(output)
            rows = [r for r in rows if not (r['model'] == family and r.get('order') == index)]
            rows.append(row)
            (results / 'measurements.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    summary = {}
    for family in MODELS:
        summary[family] = {}
        for impl in ['default', 'gfx1151_v1']:
            selected = [r for r in rows if r['model'] == family and r['status'] == 'passed' and r['implementation'] == impl]
            if selected:
                summary[family][impl] = dict(rtf=sum(r['rtf'] for r in selected) / len(selected), runs=len(selected))
    (results / 'summary.json').write_text(json.dumps(summary, indent=2))
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
        output.write('```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    if any(r['status'] != 'passed' for r in rows):
        raise SystemExit('Some inference cases failed; see measurements.json and per-case logs')


if __name__ == '__main__':
    main()
