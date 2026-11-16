#!/usr/bin/env python3
"""Collect smoke-test logs and outputs from an extracted rocBLAS release package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packages, source, output = args.packages.resolve(), args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows = platform.system() == 'Windows'
    env = os.environ.copy()

    def run(command, name, required=True, cwd=output):
        command = [str(value) for value in command]
        print('+', subprocess.list2cmdline(command), flush=True)
        (output / f'{name}.command.json').write_text(json.dumps(command, indent=2))
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=1800)
        print(f'{name}: exit_code={process.returncode}', flush=True)
        if required and process.returncode:
            raise RuntimeError(f'{name} failed; see {name}.log')
        return process.returncode

    archives = list(packages.glob('*-rocblas-test.zip' if windows else '*-rocblas-test.tar.gz'))
    if len(archives) != 1:
        raise RuntimeError(f'expected one rocBLAS archive, got {archives}')
    archive = archives[0]
    expected = next(line.split()[0] for line in (packages / 'SHA256SUMS').read_text().splitlines()
                    if line.split()[1] == archive.name)
    with archive.open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != expected:
            raise RuntimeError('archive checksum mismatch')
    binary = Path(os.environ['RUNNER_TEMP']) / f'rocblas-prebuilt-{os.environ["GITHUB_RUN_ID"]}'
    binary.mkdir()
    if windows:
        with zipfile.ZipFile(archive) as pack:
            pack.extractall(binary)
    else:
        with tarfile.open(archive) as pack:
            pack.extractall(binary, filter='data')
    manifest = json.loads((binary / 'BUILD.json').read_text())
    if manifest['source_sha'] != os.environ['EXPECTED_SOURCE_SHA'] or manifest.get('hip_blas') != 'rocblas':
        raise RuntimeError(f'unexpected package manifest: {manifest}')
    (output / 'BUILD.json').write_text(json.dumps(manifest, indent=2))
    cache_text = (packages / 'logs/CMakeCache.txt').read_text()
    if 'GGML_HIP_HIPBLASLT:BOOL=OFF' not in cache_text:
        raise RuntimeError('compatibility package did not disable hipBLASLt')

    # Install the end-user libraries/device extra separately from the build SDK.
    runtime = Path(os.environ['RUNNER_TEMP']) / f'rocblas-runtime-{os.environ["GITHUB_RUN_ID"]}'
    run([sys.executable, '-m', 'venv', runtime], 'create-runtime')
    python = runtime / ('Scripts/python.exe' if windows else 'bin/python')
    run([python, '-m', 'pip', 'install', '--index-url', 'https://stable.repo.amd.com/rocm/whl-next/',
         'rocm[libraries,device-gfx1151]==10.0.0'], 'install-runtime')
    discover = "import json,rocm_sdk; print(json.dumps(list(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries('amdhip64','hipblas','rocblas')))))"
    paths = json.loads(subprocess.check_output([python, '-c', discover], text=True))
    (output / 'runtime-paths.json').write_text(json.dumps(paths, indent=2))
    env['PATH' if windows else 'LD_LIBRARY_PATH'] = os.pathsep.join(paths) + (os.pathsep + env['PATH'] if windows else '')
    for key in ('CMAKE_PREFIX_PATH', 'HIP_PATH', 'ROCM_PATH', 'GGML_BACKEND_PATH'):
        env.pop(key, None)

    # Reuse the persistent runner-local SHA-256 model cache; no weight artifacts.
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'audio-cpp-ci/models'
    cache.mkdir(parents=True, exist_ok=True)
    host = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    models = [
        ('asr', '490fefbe91b33e1a0aeb988b97b983d3b95ecd225195e3ca5025ed00923e94d2',
         'audio-cpp/audio.cpp-gguf/resolve/7a103b33f04120535bdfe97b5f65b79d3c90d708/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-f16.gguf'),
        ('reuse', '81f47a42fe9e67e123b9a2c16803193d4d28438432ef4a0a063692615f0913f3',
         'audio-cpp/RE-USE-GGUF/resolve/f3edd063e665993410f277b7c72635dae56326e6/reuse-nemotronlabs-f32.gguf'),
    ]
    weights = {}
    for name, digest, path in models:
        target = cache / f'{digest}.gguf'
        valid = False
        if target.is_file():
            with target.open('rb') as file:
                valid = hashlib.file_digest(file, 'sha256').hexdigest() == digest
        print(f'{name}: cache_hit={valid}', flush=True)
        if not valid:
            temporary = target.with_suffix('.tmp')
            run(['curl', '--fail', '--location', '--retry', '3', '--max-time', '1500',
                 '--output', temporary, f'{host}/{path}'], f'download-{name}')
            with temporary.open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() != digest:
                    raise RuntimeError(f'{name} checksum mismatch')
            temporary.replace(target)
        weights[name] = target
    audio = output / 'input.wav'
    shutil.copyfile(source / 'assets/resources/a.wav', audio)
    suffix = '.exe' if windows else ''
    cli = binary / f'audiocpp_cli{suffix}'
    exits = {}
    for name in ('audiocpp_cli', 'audiocpp_server', 'audiocpp_gguf'):
        exits[name] = run([binary / f'{name}{suffix}', '--help'], f'{name}-help', required=False, cwd=binary)
    exits['devices'] = run([cli, '--list-devices', '--backend', 'hip', '--log'], 'devices', required=False, cwd=binary)
    if not windows:
        exits['dependencies'] = run(['ldd', cli], 'dependencies', required=False, cwd=binary)
    exits['asr'] = run([cli, '--task', 'asr', '--family', 'qwen3_asr', '--backend', 'hip',
                        '--threads', '8', '--model', weights['asr'], '--audio', audio,
                        '--request-option', 'language=en', '--text-out', output / 'asr.txt',
                        '--log', '--log-file', output / 'asr-timing.log', '--metrics'], 'asr', required=False, cwd=binary)
    exits['reuse'] = run([cli, '--task', 's2s', '--family', 'reuse', '--backend', 'hip',
                          '--threads', '8', '--model', weights['reuse'], '--audio', audio,
                          '--out', output / 'reuse.wav', '--out-format', 'float32',
                          '--log', '--log-file', output / 'reuse-timing.log', '--metrics'], 'reuse', required=False, cwd=binary)
    (output / 'exit-codes.json').write_text(json.dumps(exits, indent=2))
    if any(exits.values()):
        raise RuntimeError(f'prebuilt checks failed: {exits}')


if __name__ == '__main__':
    main()
