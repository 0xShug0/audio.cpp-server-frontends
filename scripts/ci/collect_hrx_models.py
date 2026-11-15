#!/usr/bin/env python3
"""Collect raw three-backend runs. Interpret outputs only after download."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


MODELS = {
    'irodori': ('irodori_tts', 'audio-cpp/audio.cpp-gguf', 'ca0b203eb1bb87f333a7cfbad2d5f6672b4ccba7',
                'Irodori-TTS-v4-Small-GGUF/irodori-tts-v4.1-small-mf-q8_0.gguf', 'a450c277a29b771bde3ceb21fd753bfb9a61c93aef952e0a3b20ad8d3facbbcd'),
    'breeze': ('breeze_tts', 'audio-cpp/audio.cpp-gguf', 'ca0b203eb1bb87f333a7cfbad2d5f6672b4ccba7',
               'Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf', 'b17b89b1c21107c44e8d556ab0a856d7acd06bdd0c914e68e0e57f6e33f555a3'),
    'ming3': ('ming3_tts', 'audio-cpp/Ming-3.0-TTS-0.5B-GGUF', '2eb8d37c7f10672e69c8aa33091baa04c461b66c',
              'ming-3.0-tts-0.5b-q8_0.gguf', 'fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1'),
    'omnivoice': ('omnivoice', 'audio-cpp/audio.cpp-gguf', 'ca0b203eb1bb87f333a7cfbad2d5f6672b4ccba7',
                  'OmniVoice-GGUF/omnivoice-q8_0.gguf', '0e5a721edd1b08d9067ef66f40ded609a6b14f5a8445c703f9869736cabd350a'),
    'qwen3_tts': ('qwen3_tts', 'audio-cpp/audio.cpp-gguf', 'ca0b203eb1bb87f333a7cfbad2d5f6672b4ccba7',
                  'Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf', 'b86e03d1d2077bbaed157ab7a7fc646c3e309e69afa52c5e1c86b622b4715057'),
    'qwen3_asr': ('qwen3_asr', 'audio-cpp/audio.cpp-gguf', 'ca0b203eb1bb87f333a7cfbad2d5f6672b4ccba7',
                  'Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf', '3bc2462eca3a3a8f56abcd860d0943ffdc0546547dde55a9d08589607d6f3936'),
}
ORDER = [('a1', 'hip', ''), ('b1', 'hip', 'gfx1151_v1'), ('c1', 'hrx', ''),
         ('c2', 'hrx', ''), ('b2', 'hip', 'gfx1151_v1'), ('a2', 'hip', '')]


def run(command, stem, cwd, timeout):
    command = [str(v) for v in command]
    stem.with_suffix('.command.json').write_text(json.dumps(command, indent=2) + '\n')
    start = time.time()
    log_path = stem.with_suffix('.log')
    print(f'{stem.name}: starting (timeout {timeout}s); log={log_path}', flush=True)
    with log_path.open('wb') as log:
        p = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Stream raw output as well as saving it; no runner-side interpretation.
        follow = None
        try:
            follow = subprocess.Popen(['tail', '--pid', str(p.pid), '-n', '+1', '-f', str(log_path)])
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f'{stem.name}: process timeout; terminating process group {p.pid}', flush=True)
                code = 'timeout'
        finally:
            try:
                if p.poll() is None:
                    # Cancellation must also stop the detached CLI process group.
                    interrupting = sys.exc_info()[0] is not None
                    try:
                        os.killpg(p.pid, signal.SIGKILL if interrupting else signal.SIGTERM)
                    except ProcessLookupError:
                        pass  # The child exited between poll() and killpg().
                    try:
                        p.wait(timeout=5 if interrupting else 10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # The child exited while the timeout was raised.
                        try:
                            p.wait(timeout=5)
                        except subprocess.TimeoutExpired as error:
                            (stem.parent / 'cleanup-failed.json').write_text(
                                json.dumps(dict(pid=p.pid, command=command)) + '\n')
                            raise RuntimeError('Child survived SIGKILL; refusing further GPU measurements') from error
            finally:
                if follow is not None:
                    try:
                        follow.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        follow.terminate()
                        follow.wait(timeout=5)

    return dict(exit_code=code, started_at=start, finished_at=time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--backends', default='hip,gfx1151_v1,hrx')
    parser.add_argument('--hip-models', default='')
    parser.add_argument('--blocks', default='a1,b1,c1,c2,b2,a2')
    args = parser.parse_args()
    if list(args.results.glob('*/cleanup-failed.json')):
        raise RuntimeError('Previous child cleanup failed; no further GPU measurements are safe on this runner')
    source, results = args.source.resolve(), (args.results / args.model).resolve()
    results.mkdir(parents=True, exist_ok=True)
    family, repo, revision, filename, expected_sha256 = MODELS[args.model]
    model_dir = Path(os.environ['RUNNER_TEMP']) / 'hrx-models' / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / Path(filename).name
    status = dict(family=family, repo=repo, revision=revision, file=filename, runs=[])
    try:
        cache_root = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'audio-cpp-ci'
        cache_root.mkdir(parents=True, exist_ok=True)
        cached = cache_root / expected_sha256
        status['model_sha256'] = expected_sha256
        status['cache_hit'] = False
        if cached.exists():
            with cached.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual == expected_sha256:
                status['cache_hit'] = True
            else:
                status['invalid_cache_sha256'] = actual
                print(f'{args.model}: rejecting corrupt runner cache {cached}: {actual}', flush=True)
                cached.unlink()
        if status['cache_hit']:
            print(f'{args.model}: runner cache hit {cached}', flush=True)
            (results / 'download.log').write_text(f'Runner cache hit: {cached}\nSHA-256 verified: {expected_sha256}\n')
            status['download'] = dict(exit_code=0, cache_hit=True)
        else:
            official = 'https://huggingface.co'
            endpoint = os.environ.get('HF_ENDPOINT', official).rstrip('/')
            endpoints = [endpoint] + ([official] if endpoint != official else [])
            partial = cache_root / f'{expected_sha256}.{os.getpid()}.part'
            try:
                for index, host in enumerate(endpoints):
                    url = f'{host}/{repo}/resolve/{revision}/{filename}?download=true'
                    stem = results / ('download' if host == official else 'download-mirror')
                    download = run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                                    '--max-time', '1800', '--output', partial, url], stem, source, 1900)
                    status['download'] = download
                    if download['exit_code'] == 0:
                        with partial.open('rb') as handle:
                            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                        if actual != expected_sha256:
                            raise RuntimeError(f'model SHA-256 mismatch: expected {expected_sha256}, got {actual}')
                        partial.replace(cached)
                        break
                    if index + 1 == len(endpoints):
                        raise RuntimeError('download failed; see download logs')
                    print(f'{args.model}: {host} download failed; retrying official Hugging Face', flush=True)
            finally:
                if partial.exists():
                    partial.unlink()
        shutil.copyfile(cached, model)
        status['model_bytes'] = model.stat().st_size
        en = ('The little bookshop beside the station stays open until sunset. After work, '
              'I often stop there to read a few pages and speak with the owner. On a rainy '
              'evening, I found a photograph of a white lighthouse hidden inside an old travel journal.')
        ja = ('今日は穏やかな朝です。窓を開けると、庭から花の香りが入ってきました。'
              '遠くで鳥が鳴いています。温かいお茶を飲みながら、今日の予定をゆっくり考えました。')
        if args.model == 'qwen3_asr':
            requests = [dict(id='warmup', audio=str(source / 'assets/resources/a.wav'), language='en'),
                        dict(id='measured_long', audio=str(source / 'assets/resources/speech.wav'), language='en'),
                        dict(id='measured_other', audio=str(source / 'assets/resources/qwen3_tts_longform_asr_input.wav'), language='en')]
        else:
            options = {}
            if args.model == 'irodori':
                options['no_ref'] = True
            if args.model in ('breeze', 'qwen3_tts'):
                options['instruct'] = 'A clear warm female narrator with a natural speaking pace.'
            requests = [dict(id='warmup', text='こんにちは。' if args.model == 'irodori' else 'Good morning. Welcome to the demonstration.',
                             language='ja' if args.model == 'irodori' else 'en', seed=1234, options=options),
                        dict(id='measured', text=ja if args.model == 'irodori' else en,
                             language='ja' if args.model == 'irodori' else 'en', seed=2026, options=options),
                        dict(id='repeat', text=ja if args.model == 'irodori' else en,
                             language='ja' if args.model == 'irodori' else 'en', seed=2026, options=options)]
        if args.model == 'breeze':
            for request in requests:
                request.pop('language', None)
        sequence = results / 'requests.json'
        sequence.write_text(json.dumps(requests, ensure_ascii=False, indent=2) + '\n')
        selected = set(args.backends.split(','))
        if args.model in args.hip_models.split(','):
            selected.update(['hip', 'gfx1151_v1'])
        failed_backends = set()
        for label, backend, impl in ORDER:
            if label not in args.blocks.split(',') or (impl or backend) not in selected:
                continue
            status_path = results / f'{label}.status.json'
            if status_path.exists() and json.loads(status_path.read_text())['exit_code'] == 0:
                continue
            if (backend, impl) in failed_backends:
                print(f'{args.model} {label}: skipped after this backend already failed; see its first log', flush=True)
                continue
            command = [source / 'build/debug/bin/audiocpp_cli', '--family', family,
                       '--task', 'asr' if args.model == 'qwen3_asr' else 'tts', '--mode', 'offline',
                       '--model', model, '--backend', backend, '--threads', '8', '--log', '--metrics',
                       '--request-sequence', sequence, '--out-dir', results / label]
            if impl:
                command += ['--backend-impl', impl]
            record = run(command, results / label, source, 3600 if args.model == 'qwen3_asr' else 1200)
            record.update(backend=backend, implementation=impl or 'default')
            status_path.write_text(json.dumps(record, indent=2) + '\n')
            status['runs'].append(record)
            if record['exit_code'] != 0:
                failed_backends.add((backend, impl))
            print(f'{args.model} {label}: {record["exit_code"]}', flush=True)
    finally:
        (results / 'collection.json').write_text(json.dumps(status, indent=2) + '\n')
    return 0 if all(x['exit_code'] == 0 for x in status['runs']) else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    raise SystemExit(main())
