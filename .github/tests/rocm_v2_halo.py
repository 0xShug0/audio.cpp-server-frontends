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


def operator_manifest(root):
    plan = qwen_manifest(root)
    health = plan['builds'][0]
    health['commands'].insert(0, ['cmake', '-E', 'make_directory', 'build/debug'])
    config = plan['builds'][1]['commands'][0]
    config[config.index('-S') + 1] = 'external/ggml'
    config = [x for x in config if not x.startswith(('-DENGINE_', '-DAUDIOCPP_'))]
    config += ['-DGGML_HIP=ON', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
               '-DGGML_BUILD_TESTS=ON', '-DGGML_BUILD_EXAMPLES=OFF']
    build = dict(id='operators', build_type='Debug', timeout_s=1800, commands=[config,
        ['cmake', '--build', 'build/debug', '--target', 'test-backend-ops', '--parallel', '8']])
    probe = ['build/debug/bin/test-backend-ops']
    groups = [('copy', ['-o', 'CPY']), ('rms', ['-o', 'RMS_NORM']),
              ('rms-fused', ['-o', 'RMS_NORM_MUL']),
              ('matvec', ['-o', 'MUL_MAT', '-p', 'type_a=q8_0']),
              ('matvec-fused', ['-o', 'MUL_MAT_VEC_FUSION'])]
    checks = [dict(id=impl + '-' + name,
        argv=probe + ['test', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', impl] + extra)
        for impl in ['default', 'halo'] for name, extra in groups]
    timed = []
    # Full-graph fusion cases are correctness-only: the generic perf harness repeats only the final node.
    for repeat in range(2):
        for slot, impl in enumerate(['default', 'halo', 'halo', 'default']):
            for name, extra in groups:
                if 'fused' in name:
                    continue
                timed.append(dict(id=f'r{repeat}-s{slot}-{impl}-{name}',
                    argv=probe + ['perf', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', impl] + extra))
    return dict(version=1, builds=[health, build], health_check=plan['health_check'], suites=[
        dict(id='correctness', build='operators', gpu=True, timeout_s=300, cases=checks),
        dict(id='framework', build='operators', gpu=True, timeout_s=150, cases=timed)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--operators', action='store_true')
    args = parser.parse_args()
    (args.root / 'campaign.json').write_text(json.dumps(operator_manifest(args.root) if args.operators else manifest(args.root), indent=2) + '\n')
