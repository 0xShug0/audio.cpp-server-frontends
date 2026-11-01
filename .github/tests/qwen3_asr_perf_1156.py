"""Prepare PR #1156's three-mode HIP comparison for the existing serial campaign."""
import argparse
import hashlib
import json
from pathlib import Path
from rocm_v2_qwen import manifest as qwen_manifest


def prepare(root):
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    models = []
    for size, length, digest in [
        ('0.6', 1151279968, '5c53bbb9a6b2ade85c16c290c560658ca18b78a23c323378528f1f945683fcbe'),
        ('1.7', 2473017600, '3bc2462eca3a3a8f56abcd860d0943ffdc0546547dde55a9d08589607d6f3936'),
    ]:
        models.append(dict(family='qwen3_asr', repo='audio-cpp/audio.cpp-gguf',
            revision='874fcb41ececda20bcbc0782d6b4ea7f7d069ea8',
            file=f'Qwen3-ASR-{size}B-GGUF/qwen3-asr-{size}b-q8_0.gguf',
            size=length, sha256=digest, destination=f'inputs/qwen3-asr-{size}b-q8_0.gguf'))
    (inputs / 'model-provenance.json').write_text(json.dumps(models, indent=2) + '\n')
    # Paths resolve from the native CLI's source-root working directory.
    requests = {'requests': [
        dict(id='warmup', audio='assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav'),
        dict(id='long_327', audio='assets/resources/qwen3_tts_longform_asr_input.wav'),
        dict(id='long_167', audio='assets/resources/vibevoice_two_speaker_longform.wav'),
    ]}
    request_path = inputs / 'qwen3-asr-requests.json'
    request_path.write_text(json.dumps(requests, indent=2) + '\n')
    hashes = [m['sha256'] + '  ' + m['destination'] for m in models]
    for path in [request_path] + [root / r['audio'] for r in requests['requests']]:
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(root).as_posix())
    (inputs / 'input-sha256.txt').write_text('\n'.join(hashes) + '\n')
    plan = qwen_manifest(root)
    build = plan['builds'][1]
    build['commands'][0] = [arg.replace('-DAUDIOCPP_MODELS=qwen3_tts', '-DAUDIOCPP_MODELS=qwen3_asr')
                            for arg in build['commands'][0]]
    cases = []
    for size in ['0.6', '1.7']:
        for slot, mode in enumerate(['default', 'preset', 'kernel', 'kernel', 'preset', 'default']):
            name = f'q8-{size}-{slot}-{mode}'
            argv = ['build/debug/bin/audiocpp_cli', '--family', 'qwen3_asr', '--task', 'asr',
                    '--model', f'inputs/qwen3-asr-{size}b-q8_0.gguf', '--backend', 'hip',
                    '--device', '0', '--threads', '8', '--request-sequence', 'inputs/qwen3-asr-requests.json',
                    '--out-dir', '../results/transcripts/' + name, '--log', '--metrics']
            if mode != 'default':
                argv += ['--session-option', 'qwen3_asr.perf_mode=cuda_perf']
            if mode == 'kernel':
                argv += ['--backend-impl', 'gfx1151_v1']
            cases.append(dict(id=name, argv=argv))
    plan['suites'] = [dict(id='correctness', build='model', gpu=True, timeout_s=600, cases=cases)]
    (root / 'campaign.json').write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve())
