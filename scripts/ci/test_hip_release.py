#!/usr/bin/env python3
"""Test downloaded HIP archives outside a checkout/build environment."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import wave
import zipfile


def main():
    root = Path.cwd()
    results = root / 'results'
    results.mkdir(exist_ok=True)
    windows = sys.platform == 'win32'
    executable = '.exe' if windows else ''
    for directory in (root / 'packages', root / 'inputs'):
        for line in (directory / 'SHA256SUMS').read_text().splitlines():
            expected, name = line.split(None, 1)
            with (directory / name.strip()).open('rb') as file:
                assert hashlib.file_digest(file, 'sha256').hexdigest() == expected, name
    env = os.environ.copy()
    # A fresh runtime-only installation, separate from the builder's SDK.
    runtime = root / 'runtime'
    subprocess.run([sys.executable, '-m', 'venv', runtime], check=True)
    python = runtime / ('Scripts/python.exe' if windows else 'bin/python')
    with (results / 'runtime-install.log').open('w') as log:
        subprocess.run([python, '-m', 'pip', 'install', '--index-url',
                        'https://stable.repo.amd.com/rocm/whl-next/',
                        'rocm[libraries]==10.0.0'], stdout=log, stderr=subprocess.STDOUT, check=True)
    sdk = runtime / ('Scripts/rocm-sdk.exe' if windows else 'bin/rocm-sdk')
    rocm_root = subprocess.check_output([sdk, 'path', '--root'], text=True).strip()
    rocm_bin = subprocess.check_output([sdk, 'path', '--bin'], text=True).strip()
    env['PATH'] = rocm_bin + os.pathsep + env['PATH']
    env.pop('CMAKE_PREFIX_PATH', None)
    env.pop('HIP_PATH', None)
    env.pop('ROCM_PATH', None)
    if not windows:
        env['LD_LIBRARY_PATH'] = rocm_root + '/lib'
    (results / 'runtime.json').write_text(json.dumps(dict(root=rocm_root, bin=rocm_bin), indent=2))
    scratch = Path(tempfile.mkdtemp(prefix='audiocpp-user-'))
    binary = scratch / 'binary'
    library = scratch / 'library'
    for kind, target in (('bin', binary), ('lib', library)):
        target.mkdir()
        archive, = (root / 'packages').glob(f'*-{kind}-*' + ('.zip' if windows else '.tar.gz'))
        if windows:
            with zipfile.ZipFile(archive) as pack:
                pack.extractall(target)
        else:
            with tarfile.open(archive) as pack:
                pack.extractall(target, filter='data')
        manifest = json.loads((target / 'BUILD.json').read_text())
        assert manifest['source_sha'] == os.environ['EXPECTED_SOURCE_SHA'], manifest
        assert 'gfx1151' in manifest['gpu_targets'], manifest
        (results / f'{kind}-BUILD.json').write_text(json.dumps(manifest, indent=2))
    cli = binary / ('audiocpp_cli' + executable)

    def run(command, name, timeout=300):
        print('+', subprocess.list2cmdline([str(x) for x in command]), flush=True)
        with (results / (name + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=binary, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        text = (results / (name + '.log')).read_text(errors='replace')
        print(text[-6000:], flush=True)
        return text

    devices = run([cli, '--list-devices'], 'devices')
    assert re.search(r'ROCm|HIP|Radeon', devices, re.I), devices
    if not windows:
        for path in (cli, binary / 'libggml-hip.so', library / 'libs/libaudiocpp.so'):
            dependencies = run(['ldd', path], 'ldd-' + path.name)
            assert 'not found' not in dependencies, dependencies
    moonshine = root / 'inputs/moonshine-tiny-q8_0.gguf'
    asr = [cli, '--task', 'asr', '--family', 'moonshine_asr', '--model', moonshine,
           '--backend', 'hip', '--threads', '4', '--log', '--metrics']
    run(asr + ['--audio', root / 'inputs/jfk.wav', '--text-out', results / 'jfk.txt'], 'jfk')

    def wer(reference, hypothesis):
        ref = re.findall(r'[a-z]+', reference.lower())
        hyp = re.findall(r'[a-z]+', hypothesis.lower())
        row = list(range(len(hyp) + 1))
        for i, word in enumerate(ref, 1):
            next_row = [i]
            for j, other in enumerate(hyp, 1):
                next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j-1] + (word != other)))
            row = next_row
        return row[-1] / len(ref)

    expected_jfk = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
    jfk_wer = wer(expected_jfk, (results / 'jfk.txt').read_text())
    assert jfk_wer <= 0.10, jfk_wer
    qwen = root / 'inputs/qwen3-customvoice-q8_0.gguf'
    url = 'https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/6f45df32f8fb283e4cfce3d3e1cb7688986fcbe6/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf?download=true'
    download_start = time.monotonic()
    urllib.request.urlretrieve(url, qwen)
    with qwen.open('rb') as file:
        assert hashlib.file_digest(file, 'sha256').hexdigest() == 'c8ac3f94b6e73f137c0e7ecc9270bc8faec7a476ff799b87e48844d1d0987ac3'
    download_sec = time.monotonic() - download_start
    text = 'The morning train arrives at the quiet station. Please bring your ticket and meet us beside the blue clock.'
    run([cli, '--task', 'tts', '--family', 'qwen3_tts', '--model', qwen,
         '--backend', 'hip', '--threads', '4', '--voice-id', 'ryan', '--language', 'English',
         '--text', text, '--request-option', 'seed=1234', '--out', results / 'qwen.wav',
         '--log', '--metrics'], 'qwen', timeout=600)
    with wave.open(str(results / 'qwen.wav')) as audio:
        duration = audio.getnframes() / audio.getframerate()
        assert 3 < duration < 30, duration
        assert any(audio.readframes(audio.getnframes())), 'silent output'
    run(asr + ['--audio', results / 'qwen.wav', '--text-out', results / 'qwen-asr.txt'], 'qwen-asr')
    tts_wer = wer(text, (results / 'qwen-asr.txt').read_text())
    assert tts_wer <= 0.20, tts_wer
    # Load only the library archive, not libraries in the CLI package.
    if windows:
        dll_dir = os.add_dll_directory(str(library / 'libs'))
        runtime_dir = os.add_dll_directory(rocm_bin)
    else:
        dll_dir = runtime_dir = None
    lib = ctypes.CDLL(str(library / 'libs' / ('audiocpp.dll' if windows else 'libaudiocpp.so')))
    lib.audiocpp_abi_version.restype = ctypes.c_uint32
    assert lib.audiocpp_abi_version() > 0
    registry = ctypes.c_void_p()
    lib.audiocpp_registry_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    assert lib.audiocpp_registry_create(None, ctypes.byref(registry)) == 0
    lib.audiocpp_registry_family_count.argtypes = [ctypes.c_void_p]
    lib.audiocpp_registry_family_count.restype = ctypes.c_size_t
    families = lib.audiocpp_registry_family_count(registry)
    assert families > 50, families
    lib.audiocpp_registry_free.argtypes = [ctypes.c_void_p]
    lib.audiocpp_registry_free(registry)
    # Test native UI/management startup from the extracted bundle.
    with (results / 'server.log').open('w') as log:
        server = subprocess.Popen([str(binary / ('audiocpp_server' + executable)),
                                   '--ui', '--ui-management', '--backend', 'hip', '--port', '18080', '--log'],
                                  cwd=binary, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError('server exited; see server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18080/', timeout=2) as response:
                        html = response.read()
                    assert b'<html' in html.lower() or b'<!doctype' in html.lower()
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('server UI did not start')
        finally:
            server.terminate()
            server.wait(timeout=20)
    summary = dict(jfk_wer=jfk_wer, tts_wer=tts_wer, audio_sec=duration,
                   qwen_download_sec=download_sec, c_api_families=families,
                   server_ui=True, source_sha=manifest['source_sha'])
    (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a') as file:
            file.write('```json\n' + json.dumps(summary, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()
