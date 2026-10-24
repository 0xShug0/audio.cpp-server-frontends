#!/usr/bin/env python3
"""Test downloaded HIP archives outside a checkout/build environment."""

import ctypes
import array
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


def c_api_test(library, model, audio, output):
    windows = sys.platform == 'win32'
    if windows:
        dll_dir = os.add_dll_directory(str(library))
        runtime_dir = os.add_dll_directory(os.environ['HIP_RUNTIME_BIN'])
    lib = ctypes.CDLL(str(library / ('audiocpp.dll' if windows else 'libaudiocpp.so')))
    pointer = ctypes.c_void_p
    out_pointer = ctypes.POINTER(pointer)

    def bind(name, arguments, result=ctypes.c_int):
        function = getattr(lib, 'audiocpp_' + name)
        function.argtypes = arguments
        function.restype = result
        return function

    error = bind('last_error', [], ctypes.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(error().decode())

    class Backend(ctypes.Structure):
        _fields_ = [('backend', ctypes.c_char_p), ('device', ctypes.c_int), ('threads', ctypes.c_int)]

    class ModelConfig(ctypes.Structure):
        _fields_ = [(key, ctypes.c_char_p) for key in
                    ('family_hint', 'config_id', 'weight_id', 'model_spec_override')]

    registry, loaded, session, result = pointer(), pointer(), pointer(), pointer()
    request = bind('request_create', [], pointer)()
    try:
        check(bind('registry_create', [ctypes.c_char_p, out_pointer])(None, ctypes.byref(registry)))
        families = bind('registry_family_count', [pointer], ctypes.c_size_t)(registry)
        assert families > 50, families
        config = ModelConfig(b'moonshine_asr', None, None, None)
        check(bind('model_load', [pointer, ctypes.c_char_p, ctypes.POINTER(ModelConfig), pointer, out_pointer])(
            registry, os.fsencode(model), ctypes.byref(config), None, ctypes.byref(loaded)))
        backend = Backend(b'hip', 0, 4)
        check(bind('session_create', [pointer, ctypes.c_char_p, ctypes.c_char_p,
                                     ctypes.POINTER(Backend), pointer, out_pointer])(
            loaded, b'asr', b'offline', ctypes.byref(backend), None, ctypes.byref(session)))
        with wave.open(str(audio)) as wav:
            assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
            samples = array.array('h', wav.readframes(wav.getnframes()))
            pcm = (ctypes.c_float * len(samples))(*(x / 32768 for x in samples))
            check(bind('request_set_audio', [pointer, ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_size_t, ctypes.c_int, ctypes.c_int])(
                request, pcm, len(samples), wav.getframerate(), 1))
        check(bind('session_run', [pointer, pointer, out_pointer])(session, request, ctypes.byref(result)))
        text = ctypes.c_char_p()
        check(bind('result_text', [pointer, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)])(
            result, ctypes.byref(text), None))
        output.write_text(text.value.decode(), encoding='utf-8')
        print(f'C API: {families} families; ASR: {text.value.decode()}', flush=True)
    finally:
        for name, handle in [('result', result), ('request', request), ('session', session),
                             ('model', loaded), ('registry', registry)]:
            bind(name + '_free', [pointer], None)(handle)


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
        subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', '--index-url',
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

    def run(command, name, timeout=300, cwd=None):
        print('+', subprocess.list2cmdline([str(x) for x in command]), flush=True)
        with (results / (name + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=cwd or binary, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        text = (results / (name + '.log')).read_text(errors='replace')
        print(text[-6000:], flush=True)
        return text

    devices = run([cli, '--list-devices'], 'devices')
    assert re.search(r'ROCm|HIP|Radeon', devices, re.I), devices
    if not windows:
        for path in (cli, library / 'libs/libaudiocpp.so'):
            dependencies = run(['ldd', path], 'ldd-' + path.name)
            assert 'not found' not in dependencies, dependencies
    moonshine = root / 'inputs/moonshine-tiny-q8_0.gguf'
    asr = [cli, '--task', 'asr', '--family', 'moonshine_asr', '--model', moonshine,
           '--backend', 'hip', '--threads', '4', '--log', '--metrics']
    jfk_log = run(asr + ['--audio', root / 'inputs/jfk.wav', '--text-out', results / 'jfk.txt'], 'jfk')

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
    with (results / 'qwen-download.log').open('w') as log:
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--max-time', '300',
                        '--output', str(qwen), url], stdout=log, stderr=subprocess.STDOUT, check=True)
    with qwen.open('rb') as file:
        assert hashlib.file_digest(file, 'sha256').hexdigest() == 'c8ac3f94b6e73f137c0e7ecc9270bc8faec7a476ff799b87e48844d1d0987ac3'
    download_sec = time.monotonic() - download_start
    text = 'The morning train arrives at the quiet station. Please bring your ticket and meet us beside the blue clock.'
    qwen_log = run([cli, '--task', 'tts', '--family', 'qwen3_tts', '--model', qwen,
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
    # A separate process loads only the library package and its runtime.
    env['HIP_RUNTIME_BIN'] = rocm_bin
    env['GGML_BACKEND_PATH'] = str(library / 'libs')
    if not windows:
        env['LD_LIBRARY_PATH'] = str(library / 'libs') + ':' + rocm_root + '/lib'
    run([sys.executable, Path(__file__).resolve(), '--c-api', library / 'libs', moonshine,
         root / 'inputs/jfk.wav', results / 'c-api-jfk.txt'], 'c-api', cwd=library / 'libs')
    c_api_wer = wer(expected_jfk, (results / 'c-api-jfk.txt').read_text())
    assert c_api_wer <= 0.10, c_api_wer
    env.pop('GGML_BACKEND_PATH')
    if not windows:
        env['LD_LIBRARY_PATH'] = rocm_root + '/lib'
    # Exercise the server with an ordinary model configuration and HTTP request.
    config = results / 'server.json'
    config.write_text(json.dumps(dict(host='127.0.0.1', port=18080, backend='hip', threads=4,
                                     lazy_load=True, ui=True, ui_management=True,
                                     models=[dict(id='moonshine', family='moonshine_asr',
                                                  path=str(moonshine), task='asr', mode='offline')]), indent=2))
    with (results / 'server.log').open('w') as log:
        server = subprocess.Popen([str(binary / ('audiocpp_server' + executable)),
                                   '--config', str(config), '--log'],
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
            request = urllib.request.Request('http://127.0.0.1:18080/v1/audio/transcriptions',
                                             data=json.dumps(dict(model='moonshine', audio=str(root / 'inputs/jfk.wav'))).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                transcription = json.load(response)
            (results / 'server-response.json').write_text(json.dumps(transcription, indent=2))
            server_wer = wer(expected_jfk, transcription['text'])
            assert server_wer <= 0.10, transcription
        finally:
            server.terminate()
            server.wait(timeout=20)
    jfk_rtf = float(re.search(r'^metrics\.rtf=(.+)$', jfk_log, re.M)[1])
    qwen_rtf = float(re.search(r'^metrics\.rtf=(.+)$', qwen_log, re.M)[1])
    summary = dict(jfk_wer=jfk_wer, jfk_rtf=jfk_rtf, tts_wer=tts_wer, qwen_rtf=qwen_rtf, audio_sec=duration,
                   qwen_download_sec=download_sec, c_api_wer=c_api_wer,
                   server_ui=True, server_wer=server_wer, source_sha=manifest['source_sha'])
    (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a') as file:
            file.write('```json\n' + json.dumps(summary, indent=2) + '\n```\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--c-api':
        c_api_test(*(Path(x) for x in sys.argv[2:]))
    else:
        main()
