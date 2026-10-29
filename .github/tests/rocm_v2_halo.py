"""Native Halo implementation selection and operator/model checks. No environment setup."""
import argparse
import json
from pathlib import Path
from rocm_v2_qwen import manifest as qwen_manifest


def manifest(root):
    plan = qwen_manifest(root)
    model = next(item for item in plan['builds'] if item['id'] == 'model')
    model['commands'][1] = ['cmake', '--build', 'build/debug', '--target',
        'audiocpp_cli', 'halo_backend_selection', 'halo_backend_ops', '--parallel', '8']
    for suite in plan['suites']:
        for case in suite['cases']:
            argv = case['argv']
            if '--backend' not in argv:
                continue
            index = argv.index('--backend')
            implementation = 'halo' if argv[index + 1] == 'rocm_v2' else 'default'
            argv[index + 1] = 'hip'
            argv[index + 2:index + 2] = ['--backend-impl', implementation]
            if suite['id'] == 'framework':
                case['id'] = implementation
                argv[:] = [arg.replace('profile/rocm_v2', 'profile/halo') for arg in argv]
    plan['suites'][-1]['cases'][0]['argv'] += ['--implementation', 'halo']
    probe = ['build/debug/bin/halo_backend_ops', 'test', '-b', 'ROCm0', '--suite', 'halo']
    cases = [dict(id='selection', argv=['build/debug/bin/halo_backend_selection', '--log'])]
    for impl in ['default', 'halo']:
        for kind, extra in [('f32', ['-o', 'MUL_MAT', '-p', 'prec=f32']),
                            ('default-precision', ['-o', 'MUL_MAT', '-p', 'prec=def']),
                            ('im2col', ['-o', 'IM2COL']),
                            ('inherited-types', ['-o', 'MUL_MAT', '-p', 'type_a=(f16|bf16|q8_0|q4_0)'])]:
            cases.append(dict(id=impl + '-' + kind, argv=probe + ['--backend-impl', impl] + extra))
    plan['suites'].insert(0, dict(id='operators', build='model', gpu=True, timeout_s=240, cases=cases))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    (args.root / 'campaign.json').write_text(json.dumps(manifest(args.root), indent=2) + '\n')
