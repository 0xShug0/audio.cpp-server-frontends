#!/usr/bin/env python3
"""Collect raw prebuilt HIP A/B evidence. No log/audio analysis runs here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ORDER = [('a1', None), ('b1', 'gfx1151_v1'), ('b2', 'gfx1151_v1'), ('a2', None)]
TEXT = ('The community library will reopen next Monday after several months of renovation. '
        'Visitors will find a larger reading room and quiet desks near the windows. '
        'Please ask a member of staff if you need help finding a book.')
REFERENCE_TEXT = ("Some call me nature. Others call me Mother Nature. I've been here for over "
                  '4.5 billion years. 22,500 times longer than you.')
LYRICS = ('[Verse]\nSoft morning light is touching the window.\nI hear the city waking below.\n'
          'The river carries a silver reflection.\nWe follow the road where the wildflowers grow.\n'
          '[Pre-Chorus]\nEvery footstep finds a rhythm.\nEvery new day brings a song.\n'
          '[Chorus]\nStay with the rhythm, let it carry us home.\nSing with the sunrise, we are never alone.\n'
          'Lift up your voice and remember the light.\nWe keep on walking till the stars fill the night.\n')


def requests_for(model, workspace):
    model_id = model['id']
    options = {'seed': 1234}
    request = {'text': TEXT, 'options': options}
    task = 'clon'
    session = []
    model_path = workspace / model['files'][0]['path']
    if model_id in ('chatterbox', 'fish_audio', 'higgs_audio_tts', 'index_tts2_5'):
        request['voice_ref'] = str(workspace / 'b.wav')
        request['reference_text'] = REFERENCE_TEXT
    elif model_id == 'pocket_tts':
        task = 'tts'
        options['voice_embedding_file'] = str(workspace / model['files'][1]['path'])
    elif model_id == 'vibevoice':
        task = 'tts'
        request['text'] = ('Speaker 1: The library will reopen next Monday. There is a larger reading room '
                           'and more space near the windows.\nSpeaker 2: That sounds wonderful. '
                           'I will bring the children after school so we can find some new books together.')
        options['voice_samples'] = f'{workspace / "a.wav"},{workspace / "b.wav"}'
    elif model_id == 'yue2':
        task = 'gen'
        model_path = workspace
        request['text'] = LYRICS
        options.update(style='English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix',
                       cot='off', semantic_min_tokens=192, semantic_max_tokens=640,
                       abc_min_tokens=0, abc_max_tokens=1)
        session = ['--session-option', 'yue2.model_gguf=yue2-3b-q4_0.gguf',
                   '--session-option', 'yue2.vae_gguf=yue2-vae-f16.gguf']
    elif model_id == 'ace_step_turbo':
        task = 'gen'
        request['text'] = ('Bright English indie pop with a clear warm lead vocal, acoustic guitar, '
                           'steady bass, clean drums, and a rising melodic chorus.')
        options.update(lyrics=LYRICS, duration_sec=20, num_inference_steps=8,
                       route='text2music', enable_thinking=True)
    else:
        raise ValueError(model_id)
    return task, model_path, session, [dict(request, id='warmup'), dict(request, id='measured')]


def run_raw(command, stem, env, cwd, timeout):
    command = [str(x) for x in command]
    stem.with_suffix('.command.json').write_text(json.dumps(command, indent=2) + '\n')
    started = time.time()
    with stem.with_suffix('.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 'timeout'
    return {'exit_code': code, 'started_at': started, 'finished_at': time.time()}


def collect(args):
    manifest = json.loads(args.manifest.read_text())
    model = next(x for x in manifest['models'] if x['id'] == args.model)
    results = args.results.resolve() / args.model
    status_file = results / 'status.json'
    if status_file.exists() and json.loads(status_file.read_text()).get('collection_complete'):
        print(f'{args.model}: collection already complete; not rerunning', flush=True)
        return 0
    results.mkdir(parents=True, exist_ok=True)
    workspace = args.workspace.resolve() / args.model
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    env = os.environ.copy()
    env.update(HF_HOME=str(workspace / 'hf-cache'), HF_XET_CACHE=str(workspace / 'xet-cache'),
               HF_HUB_DISABLE_PROGRESS_BARS='1', OMP_NUM_THREADS=str(manifest['threads']),
               OPENBLAS_NUM_THREADS=str(manifest['threads']))
    status = {'model': model, 'source_sha': manifest['source_sha'], 'variants': [],
              'collection_complete': False}

    def save():
        status_file.write_text(json.dumps(status, indent=2) + '\n')

    save()
    try:
        download_code = ('import json,sys; from huggingface_hub import snapshot_download; '
                         'snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],'
                         'local_dir=sys.argv[3],allow_patterns=json.loads(sys.argv[4]))')
        download = run_raw([sys.executable, '-c', download_code, model['repo'], model['revision'],
                            workspace, json.dumps([f['path'] for f in model['files']])],
                           results / 'download', env, args.binary.parent, 1800)
        status['download'] = download
        if download['exit_code'] != 0:
            raise RuntimeError('model download failed; see raw download log')
        for file in model['files']:
            with (workspace / file['path']).open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != file['sha256']:
                raise RuntimeError(f'download checksum mismatch: {file["path"]}')
        for name in ('a.wav', 'b.wav'):
            url = f'https://raw.githubusercontent.com/0xShug0/audio.cpp/{manifest["source_sha"]}/assets/resources/{name}'
            with urllib.request.urlopen(url, timeout=120) as response, (workspace / name).open('wb') as output:
                shutil.copyfileobj(response, output)
        task, model_path, session, requests = requests_for(model, workspace)
        sequence = workspace / 'requests.json'
        sequence.write_text(json.dumps(requests, indent=2) + '\n')
        shutil.copy2(sequence, results / 'requests.json')
        shutil.copy2(args.manifest, results / 'campaign-manifest.json')
        for label, implementation in ORDER:
            variant_env = env.copy()
            if env.get('AUDIOCPP_HIP_COMPARE_Q4_PAIR'):
                implementation = 'gfx1151_v1'
                if label.startswith('a'):
                    variant_env['AUDIOCPP_HIP_Q4_PAIR_CONTROL'] = '1'
                else:
                    variant_env.pop('AUDIOCPP_HIP_Q4_PAIR_CONTROL', None)
            output = results / label
            command = [args.binary, '--task', task, '--family', model['family'], '--mode', 'offline',
                       '--model', model_path, '--backend', 'hip', '--threads', manifest['threads'],
                       '--request-sequence', sequence, '--out-dir', output, '--log', '--metrics', *session]
            if implementation:
                command += ['--backend-impl', implementation]
            try:
                variant = run_raw(command, results / label, variant_env, args.binary.parent, args.timeout)
            except Exception as error:
                variant = {'exit_code': 'harness_error', 'error': str(error)}
            variant.update(label=label, implementation=implementation or 'default',
                           q4_pair_control='AUDIOCPP_HIP_Q4_PAIR_CONTROL' in variant_env,
                           outputs=[{'path': str(p.relative_to(results)), 'bytes': p.stat().st_size}
                                    for p in output.rglob('*') if p.is_file()] if output.exists() else [])
            status['variants'].append(variant)
            save()
            print(f'{args.model} {label}: {variant["exit_code"]}', flush=True)
        # This records execution completion, not correctness or performance.
        # Wav/log interpretation and acceptance happen outside the runner.
        status['collection_complete'] = all(
            v['exit_code'] == 0 and v['outputs'] for v in status['variants'])
    except Exception as error:
        status['error'] = str(error)
        print(f'{args.model}: {error}', flush=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=False)
        status['weights_and_download_cache_removed'] = not workspace.exists()
        save()
    return 0 if status['collection_complete'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    sys.exit(collect(parser.parse_args()))
