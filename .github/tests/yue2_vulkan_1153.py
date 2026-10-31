"""Prepare ten YuE2 generation commands for the existing campaign runner."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(root):
    here = Path(__file__).resolve().parent
    cases = json.loads((here / 'yue2-vulkan-1153-cases.json').read_text())
    models = json.loads((here / 'yue2-vulkan-1153-models.json').read_text())
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    requests = inputs / 'yue2-requests.json'
    requests.write_text(json.dumps(cases, indent=2) + '\n')
    (inputs / 'model-provenance.json').write_text(json.dumps(models, indent=2) + '\n')
    hashes = [r['sha256'] + '  ' + r['destination'] for r in models if r['sha256']]
    hashes.append(hashlib.sha256(requests.read_bytes()).hexdigest() + '  inputs/yue2-requests.json')
    (inputs / 'input-sha256.txt').write_text('\n'.join(hashes) + '\n')
    results = root.parent / 'results'
    for folder in ('wavs', 'timing'):
        (results / folder).mkdir(parents=True, exist_ok=True)
    commands = []
    for case in cases:
        name = f'{case["id"]:02d}-seed-{case["seed"]}'
        argv = ['build/debug/bin/audiocpp_cli', '--backend', 'vulkan', '--device', '0',
                '--threads', '8', '--family', 'yue2', '--task', 'gen', '--model', 'inputs/yue2',
                '--seed', str(case['seed']), '--session-option', 'yue2.model_gguf=yue2-3b-q4_0.gguf',
                '--session-option', 'yue2.vae_gguf=yue2-vae-f16.gguf',
                '--session-option', 'yue2.perf_mode=off', '--session-option', 'yue2.kv_cache_type=f32',
                '--session-option', 'yue2.conv_weight_type=f32',
                '--request-option', 'semantic_temperature=0.8', '--request-option', 'semantic_top_p=0.93',
                '--request-option', 'semantic_repetition_penalty=1.1', '--request-option', 'cot=off',
                '--request-option', 'style=' + case['style'], '--lyrics', '[Instrumental]',
                '--out', '../results/wavs/' + name + '.wav', '--metrics', '--log',
                '--log-file', '../results/timing/' + name + '.log']
        commands.append({'id': name, 'argv': argv})
    manifest = {'version': 1, 'builds': [{'id': 'vulkan', 'build_type': 'Debug', 'timeout_s': 1200,
        'commands': [['cmake', '-S', '.', '-B', 'build/debug', '-G', 'Ninja',
                      '-DCMAKE_BUILD_TYPE=Debug', '-DCMAKE_CXX_STANDARD=17',
                      '-DENGINE_ENABLE_CUDA=OFF', '-DENGINE_ENABLE_HIP=OFF',
                      '-DENGINE_ENABLE_VULKAN=ON', '-DENGINE_BUILD_TESTS=OFF',
                      '-DENGINE_BUILD_WARMBENCH=OFF', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF',
                      '-DAUDIOCPP_MODEL_SET=custom', '-DAUDIOCPP_MODELS=yue2',
                      '-DVulkan_LIBRARY=/lib/x86_64-linux-gnu/libvulkan.so.1'],
                     ['cmake', '--build', 'build/debug', '--target', 'audiocpp_cli', '--parallel', '8']]}],
        'suites': [{'id': 'correctness', 'build': 'vulkan', 'gpu': True, 'timeout_s': 900,
                    'cases': commands}],
        'health_check': {'argv': ['vulkaninfo', '--summary'], 'timeout_s': 20}}
    (root / 'campaign.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', required=True)
    args = parser.parse_args()
    prepare(args.root.resolve())
