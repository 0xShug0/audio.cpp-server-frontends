#!/usr/bin/env python3
"""Run prebuilt HIP cases sequentially and retain raw logs, outputs, and exit codes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import wave

MODELS = {
    'reuse': ('RE-USE-GGUF', 'f3edd063e665993410f277b7c72635dae56326e6',
              'reuse-nemotronlabs-f32.gguf', '81f47a42fe9e67e123b9a2c16803193d4d28438432ef4a0a063692615f0913f3'),
    'soulx_flashhead': ('SoulX-FlashHead-GGUF', 'cc83157045008de647a4ba11ca5ed19e66caa575',
                        'SoulX-FlashHead-Lite-Q8_0.gguf', '29f186f666a4b1ee8af257a2163d521545d94c600c46d97998c57d6fefc95751'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    assert env.get('CONDA_DEFAULT_ENV') == 'qwen3-tts', env.get('CONDA_DEFAULT_ENV')
    env['OMP_NUM_THREADS'] = '8'
    env['OPENBLAS_NUM_THREADS'] = '8'

    def run(command, name, cwd=root, required=True, timeout=1800):
        command = [str(x) for x in command]
        print('+', subprocess.list2cmdline(command), flush=True)
        (results / f'{name}.command.json').write_text(json.dumps(command, indent=2))
        with (results / f'{name}.log').open('w') as log:
            try:
                process = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        print(f'{name}: exit_code={code}', flush=True)
        if required and code != 0:
            raise RuntimeError(f'{name} failed: {code}; see {name}.log')
        return code

    def download(url, destination, expected=None):
        run(['curl', '--fail', '--location', '--retry', '3', '--max-time', '900',
             '--output', destination, url], 'download-' + destination.name)
        if expected:
            with destination.open('rb') as file:
                assert hashlib.file_digest(file, 'sha256').hexdigest() == expected, destination

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
    assert manifest['source_sha'] == args.source_sha, manifest
    (results / 'BUILD.json').write_text(json.dumps(manifest, indent=2))
    cli = binary / 'audiocpp_cli'
    run([cli, '--version'], 'version', cwd=binary)
    run([cli, '--list-devices'], 'devices', cwd=binary)

    inputs = root / 'inputs'
    inputs.mkdir()
    source_url = f'https://raw.githubusercontent.com/0xShug0/audio.cpp/{args.source_sha}/assets/resources/'
    rows = []
    for family in args.models:
        repo, revision, filename, sha256 = MODELS[family]
        model = inputs / filename
        try:
            download(f'https://huggingface.co/audio-cpp/{repo}/resolve/{revision}/{filename}', model, sha256)
            audio_name = 'vibevoice_two_speaker_longform.wav' if family == 'reuse' else 'a.wav'
            download(source_url + audio_name, inputs / audio_name)
            if family == 'reuse':
                audio = inputs / 'reuse-60s.wav'
                with wave.open(str(inputs / audio_name)) as source:
                    with wave.open(str(audio), 'wb') as dest:
                        dest.setparams(source.getparams())
                        dest.writeframes(source.readframes(source.getframerate() * 60))
                options = {}
            else:
                audio = inputs / audio_name
                download(f'https://huggingface.co/audio-cpp/{repo}/resolve/{revision}/demo/portrait.png', inputs / 'portrait.png')
                options = {'reference_image_file': str(inputs / 'portrait.png'), 'seed': '42',
                           'num_inference_steps': '4', 'height': '512', 'width': '512'}
            sequence = inputs / (family + '-requests.json')
            sequence.write_text(json.dumps([{'id': name, 'audio': str(audio), 'options': options}
                                            for name in ['warmup', 'measured']], indent=2))
            (results / (family + '-requests.json')).write_text(sequence.read_text())
        except Exception as error:
            rows.append(dict(model=family, phase='setup', error=str(error)))
            (results / 'exit-codes.json').write_text(json.dumps(rows, indent=2))
            print(f'{family} setup failed: {error}', flush=True)
            continue
        for index, impl in enumerate(['default', 'gfx1151_v1', 'gfx1151_v1', 'default']):
            name = f'{family}-{index}-{impl}'
            command = [cli, '--family', family, '--task', 's2s' if family == 'reuse' else 'gen',
                       '--mode', 'offline', '--model', model, '--backend', 'hip', '--device', '0',
                       '--threads', '8', '--request-sequence', sequence, '--out-dir', results / name, '--log']
            if impl != 'default':
                command += ['--backend-impl', impl]
            try:
                code = run(command, name, cwd=binary, required=False)
                rows.append(dict(model=family, implementation=impl, order=index, exit_code=code))
            except Exception as error:
                rows.append(dict(model=family, implementation=impl, order=index, error=str(error)))
            (results / 'exit-codes.json').write_text(json.dumps(rows, indent=2))
    print('Collection complete. Inspect exit-codes.json, logs, and outputs locally.', flush=True)


if __name__ == '__main__':
    main()
