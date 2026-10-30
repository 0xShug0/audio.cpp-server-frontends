"""Native Halo implementation selection and operator/model checks. No environment setup."""
import argparse
import array
import math
import wave
import csv
import hashlib
import subprocess
import shutil
import urllib.request
import re
import statistics
import json
from pathlib import Path
from rocm_v2_qwen import manifest as qwen_manifest, REQUESTS as QWEN_REQUESTS


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


def followup_manifest(root):
    plan = qwen_manifest(root)
    build = plan['builds'][1]
    build['id'] = 'operators'
    build['commands'][0] = [x.replace('-DAUDIOCPP_MODELS=qwen3_tts', '-DAUDIOCPP_MODELS=qwen3_tts,irodori_tts,yue2')
                            for x in build['commands'][0]]
    build['commands'][1] = ['cmake', '--build', 'build/debug', '--target', 'halo_backend_ops', '--parallel', '8']
    prefix = ['build/debug/bin/halo_backend_ops']
    checks = []
    for impl in ['default', 'halo']:
        for name, extra in [('copy', ['-o', 'CPY,CONT']),
                            ('matvec', ['-o', 'MUL_MAT', '-p', 'type_a=q8_0']),
                            ('matvec-fused', ['-o', 'MUL_MAT_VEC_FUSION'])]:
            checks.append(dict(id=impl + '-' + name, argv=prefix +
                ['test', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', impl] + extra))
    timed = []
    for repeat in range(2):
        for slot, impl in enumerate(['default','halo','halo','default']):
            for name, extra in [('matvec', ['-o','MUL_MAT','-p','type_a=q8_0']),
                                ('copy', ['-o','CPY','-p',r'ne=\[(64|128),'])]:
                timed.append(dict(id=f'r{repeat}-s{slot}-{impl}-{name}', argv=prefix +
                    ['perf','-b','ROCm0','--suite','halo','--backend-impl',impl] + extra))
    timed.append(dict(id='trace', argv=['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--output-format', 'csv',
        '--output-directory', '../results/profile/q8', '--'] + prefix +
        ['test', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', 'halo',
         '-o', 'CPY,CONT,MUL_MAT,MUL_MAT_VEC_FUSION', '-p', r'ne=\[(64|128),|type_a=q8_0|type=q8_0']))
    return dict(version=1, builds=[plan['builds'][0], build], health_check=plan['health_check'], suites=[
        dict(id='correctness', build='operators', gpu=True, timeout_s=180, cases=checks),
        dict(id='framework', build='operators', gpu=True, timeout_s=120, cases=timed)])


def model_requests():
    short = '今日は穏やかな朝です。窓を開けると、庭から花の香りが入ってきました。遠くで鳥が鳴いています。温かいお茶を飲みながら、今日の予定をゆっくり考えました。'
    long = '駅前の小さな本屋には、いつも静かな音楽が流れています。私は仕事が終わると、よくその店に立ち寄ります。店主のおじいさんは本について詳しく、私が探しているものをすぐに見つけてくれます。ある雨の日、古い旅行記を手に取ると、ページの間から一枚の写真が落ちました。そこには、海に向かって建つ白い灯台が写っていました。写真の裏には、またここで会いましょう、とだけ書かれていました。私はその言葉が気になって、閉店の時間まで本を読み続けました。外に出ると雨はすっかり止み、夕焼けが濡れた道を明るく照らしていました。'
    irodori = [dict(id=name, text=text, language='ja', seed=1234, options=dict(no_ref=True))
        for name, text in [('warmup', 'こんにちは。今日はよい天気ですね。'), ('short', short),
                           ('long', long), ('short_repeat', short)]]
    lyrics = '[Verse]\nSoft morning light is touching the window.\nI hear the city waking below.\n[Chorus]\nStay with the rhythm, let it carry us home.\nSing with the sunrise, we are never alone.'
    yue2 = [dict(id=name, text=text, seed=1234, options=dict(cot='off', num_inference_steps=8,
            style='English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix'))
        for name, text in [('warmup', '[Verse]\nThe morning sun is shining.\nI hear a song outside.'),
                           ('song', lyrics), ('song_repeat', lyrics)]]
    return dict(irodori_tts=irodori, qwen3_tts=QWEN_REQUESTS, yue2=yue2)


def prepare_models(root):
    records = json.loads((root / '.github/tests/halo-model-files.json').read_text())
    for record in records:
        target = root / record['destination']
        target.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://huggingface.co/{repo}/resolve/{revision}/{file}?download=true'.format(**record)
        subprocess.run(['curl', '--fail', '--location', '--connect-timeout', '20', '--max-time', '600',
                        '--retry', '2', '-o', str(target), url], check=True)
        data_hash = hashlib.sha256()
        with target.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                data_hash.update(block)
        if target.stat().st_size != record['size']:
            raise ValueError('Size mismatch: ' + record['file'])
        digest = data_hash.hexdigest()
        if record['sha256'] and digest != record['sha256']:
            raise ValueError('SHA256 mismatch: ' + record['file'])
        if record.get('blob_id'):
            data = target.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != record['blob_id']:
                raise ValueError('Git blob mismatch: ' + record['file'])
        record['sha256'] = digest
    inputs = root / 'inputs'
    with (inputs / 'reference.wav').open('wb') as stream:
        subprocess.run(['git', 'show', 'HEAD:assets/resources/a.wav'], cwd=root, stdout=stream, check=True)
    for family, requests in model_requests().items():
        (inputs / (family + '-requests.json')).write_text(json.dumps(requests, ensure_ascii=False, indent=2) + '\n')
    (inputs / 'model-provenance.json').write_text(json.dumps(records, indent=2) + '\n')
    hashes = []
    for file in sorted(inputs.rglob('*')):
        if file.is_file() and file.name != 'input-sha256.txt':
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            hashes.append(digest + '  ' + file.relative_to(root).as_posix())
    (inputs / 'input-sha256.txt').write_text('\n'.join(hashes) + '\n')


def model_manifest(root, candidate='.', output='../results/models'):
    plan = qwen_manifest(root)
    base = plan['builds'][1]
    config = [x.replace('-DAUDIOCPP_MODELS=qwen3_tts', '-DAUDIOCPP_MODELS=qwen3_tts,irodori_tts,yue2')
              for x in base['commands'][0]]
    builds = [plan['builds'][0]]
    for variant, cwd in [('baseline', '../source-baseline'), ('candidate', candidate)]:
        builds.append(dict(id=variant, cwd=cwd, build_type='Debug', timeout_s=1800, commands=[config,
            ['cmake', '--build', 'build/debug', '--target', 'audiocpp_cli', '--parallel', '8'],
            ['ldd', 'build/debug/bin/audiocpp_cli']]))
    suites = [dict(id='correctness', build='candidate', gpu=True, timeout_s=30,
        cases=[dict(id='health', argv=['build/debug/device-health', 'health'])])]
    for family, model, extra in [
            ('irodori_tts', 'inputs/irodori.gguf', []),
            ('qwen3_tts', 'inputs/qwen.gguf', []),
            ('yue2', 'inputs/yue2', ['--session-option', 'yue2.model_gguf=yue2-3b-q4_0.gguf',
                                    '--session-option', 'yue2.vae_gguf=yue2-vae-f16.gguf'])]:
        for label, variant in [('a1', 'baseline'), ('b1', 'candidate'), ('b2', 'candidate'), ('a2', 'baseline')]:
            binary = ('../source-baseline/' if variant == 'baseline' else (candidate + '/')) + 'build/debug/bin/audiocpp_cli'
            name = family + '-' + label
            argv = [binary, '--task', 'gen' if family == 'yue2' else 'tts', '--family', family,
                    '--backend', 'hip', '--backend-impl', 'halo', '--threads', '8', '--model', model,
                    '--request-sequence', 'inputs/' + family + '-requests.json',
                    '--out-dir', output + '/wavs/' + name, '--log', '--metrics'] + extra
            suites.append(dict(id=name, build=variant, gpu=True, timeout_s=900,
                               cases=[dict(id='session', argv=argv)]))
    suites.append(dict(id='framework', gpu=False, timeout_s=180,
        cases=[dict(id='summary', argv=['python3', '.github/tests/rocm_v2_halo.py',
                                       '--root', output, '--summarize-models'])]))
    return dict(version=1, builds=builds, health_check=plan['health_check'], suites=suites)


def hip_baseline_manifest(root):
    plan = model_manifest(root, output='../results')
    plan['builds'] = [b for b in plan['builds'] if b['id'] != 'baseline']
    for suite in plan['suites']:
        if suite.get('build') == 'baseline':
            suite['build'] = 'candidate'
            for case in suite['cases']:
                argv = case['argv']
                argv[0] = './build/debug/bin/audiocpp_cli'
                index = argv.index('--backend-impl')
                del argv[index:index + 2]
    plan['suites'][-1]['cases'][0]['argv'] += ['--baseline-implementation','default']
    return plan



def irodori_main_manifest(root):
    plan = qwen_manifest(root)
    build = plan['builds'][1]
    build['commands'][0] = [x.replace('-DAUDIOCPP_MODELS=qwen3_tts', '-DAUDIOCPP_MODELS=irodori_tts')
                             for x in build['commands'][0]]
    def command(label, selected):
        argv = ['build/debug/bin/audiocpp_cli', '--task', 'tts', '--family', 'irodori_tts',
                '--backend', 'hip', '--threads', '8', '--model', 'inputs/irodori.gguf',
                '--request-sequence', 'inputs/irodori_tts-requests.json',
                '--out-dir', '../results/wavs/irodori_tts-' + label, '--log', '--metrics']
        if selected:
            argv += ['--backend-impl', 'gfx1151_v1']
        return argv
    suites = [dict(id='correctness', build='model', gpu=True, timeout_s=30,
                   cases=[dict(id='health', argv=['build/debug/device-health', 'health'])])]
    for label in ['a1', 'b1', 'b2', 'a2']:
        suites.append(dict(id='irodori_tts-' + label, build='model', gpu=True, timeout_s=600,
                           cases=[dict(id='session', argv=command(label, label.startswith('b')))]))
    suites.append(dict(id='framework', build='model', gpu=False, timeout_s=120,
        cases=[dict(id='summary', argv=['python3', '.github/tests/rocm_v2_halo.py', '--root', '../results',
            '--summarize-models', '--families', 'irodori_tts', '--baseline-implementation', 'default',
            '--candidate-implementation', 'gfx1151_v1'])]))
    return dict(version=1, builds=plan['builds'], health_check=plan['health_check'], suites=suites)


def tts_requests():
    requests = {}
    for family in ['breeze_tts', 'ming3_tts', 'omnivoice', 'qwen3_tts']:
        requests[family] = [dict(id=r['id'], text=r['text'], seed=r['seed'])
                            for r in QWEN_REQUESTS if r['id'] != 'medium']
        for request in requests[family]:
            if family != 'breeze_tts':
                request['language'] = 'English' if family == 'qwen3_tts' else 'en'
            if family == 'qwen3_tts':
                request['options'] = {'instruct': 'A calm adult narrator with a warm voice.'}
    return requests


def tts_main_manifest(root):
    plan = irodori_main_manifest(root)
    families = list(tts_requests())
    plan['builds'][1]['commands'][0] = [
        x.replace('-DAUDIOCPP_MODELS=irodori_tts', '-DAUDIOCPP_MODELS=' + ','.join(families))
        for x in plan['builds'][1]['commands'][0]]
    suites = [plan['suites'][0]]
    for family in families:
        for label in ['a1', 'b1', 'b2', 'a2']:
            name = family + '-' + label
            argv = ['build/debug/bin/audiocpp_cli', '--task', 'vdes' if family == 'qwen3_tts' else 'tts',
                    '--family', family, '--backend', 'hip', '--threads', '8',
                    '--model', 'inputs/' + family + '.gguf',
                    '--request-sequence', 'inputs/' + family + '-requests.json',
                    '--out-dir', '../results/wavs/' + name, '--log', '--metrics']
            if label.startswith('b'):
                argv += ['--backend-impl', 'gfx1151_v1']
            suites.append(dict(id=name, build='model', gpu=True, timeout_s=600,
                               cases=[dict(id='session', argv=argv)]))
    suites.append(dict(id='framework', build='model', gpu=False, timeout_s=180, cases=[
        dict(id=family, argv=['python3', '.github/tests/rocm_v2_halo.py', '--root', '../results',
            '--summarize-models', '--families', family, '--baseline-implementation', 'default',
            '--candidate-implementation', 'gfx1151_v1', '--request-set', 'tts-main']) for family in families]))
    plan['suites'] = suites
    return plan


def tts_profile_manifest(root):
    plan = tts_main_manifest(root)
    build = plan['builds'][1]
    build['commands'][0] = [x.replace('breeze_tts,ming3_tts,omnivoice,qwen3_tts',
                                      'ming3_tts,omnivoice') for x in build['commands'][0]]
    cases = []
    for family in ['ming3_tts', 'omnivoice']:
        for label in ['a1']:
            name = family + '-' + label
            suite = next(s for s in plan['suites'] if s['id'] == name)
            command = ['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--hip-runtime-trace',
                       '--collection-period', '4:1:1', '4:1:1', '--output-format', 'csv',
                       '--output-directory', '../results/profile/' + name, '--'] + suite['cases'][0]['argv']
            cases.append(dict(id=name, argv=command))
    plan['suites'] = [plan['suites'][0], dict(id='framework', build='model', gpu=True,
                                           timeout_s=300, cases=cases)]
    return plan


def breeze_ar_manifest(root):
    from copy import deepcopy
    plan = tts_main_manifest(root)
    candidate = plan['builds'][1]
    candidate['commands'][0] = [
        '-DAUDIOCPP_MODELS=breeze_tts' if x.startswith('-DAUDIOCPP_MODELS=') else x
        for x in candidate['commands'][0]]
    baseline = deepcopy(candidate)
    baseline['id'] = 'baseline'
    baseline['cwd'] = '../source-baseline'
    plan['builds'].insert(1, baseline)
    plan['suites'] = [s for s in plan['suites']
                      if s['id'] in ['correctness', 'framework'] or s['id'].startswith('breeze_tts-')]
    for suite in plan['suites']:
        if suite['id'].startswith('breeze_tts-'):
            command = suite['cases'][0]['argv']
            if suite['id'].split('-')[-1].startswith('a'):
                suite['build'] = 'baseline'
                command[0] = '../source-baseline/build/debug/bin/audiocpp_cli'
                command += ['--backend-impl', 'gfx1151_v1']
            else:
                command[command.index('--backend-impl') + 1] = 'ar_stream2'
        elif suite['id'] == 'framework':
            suite['cases'] = [c for c in suite['cases'] if c['id'] == 'breeze_tts']
            command = suite['cases'][0]['argv']
            command[command.index('--baseline-implementation') + 1] = 'gfx1151_v1'
            command[command.index('--candidate-implementation') + 1] = 'ar_stream2'
    return plan


def rdna_depth_manifest(root):
    plan = qwen_manifest(root)
    build = plan['builds'][1]
    build['commands'][0] = ['-DAUDIOCPP_MODELS=breeze_tts' if x.startswith('-DAUDIOCPP_MODELS=') else x
                              for x in build['commands'][0]]
    build['commands'][1] = ['cmake', '--build', 'build/debug', '--target',
                            'decoder_packed_projection_test', 'test-backend-ops', '--parallel', '8']
    build['commands'].append(['cmake', '-E', 'make_directory', '../results/hidden', '../results/isa'])
    implementations = ['gfx1151_v1', 'ar_nt', 'ar_stream', 'ar_stream2', 'ar_stream4']
    math = [dict(id=impl, argv=['build/debug/bin/test-backend-ops', 'test', '-b', 'ROCm0',
            '--suite', 'gfx1151_v1', '--backend-impl', impl, '-o', 'MUL_MAT',
            '-p', r'type_a=q8_0,type_b=f32,m=[0-9]+,n=(1|2),']) for impl in implementations]
    timed = []
    for batch in [1, 2]:
        for round in range(2):
            for slot, impl in enumerate(implementations + list(reversed(implementations))):
                name = f'b{batch}-r{round}-s{slot}-{impl}'
                timed.append(dict(id=name, depends_on=['correctness/' + impl],
                    argv=['build/debug/bin/decoder_packed_projection_test', '--ar-probe',
                          '--impl', impl, '--batch', str(batch), '--output', '../results/hidden/' + name + '.safetensors', '--log']))
    plan['suites'] = [
        dict(id='correctness', build='model', gpu=True, timeout_s=90, cases=math),
        dict(id='framework', build='model', gpu=True, timeout_s=180, cases=timed)]
    plan['builds'].append(dict(id='isa', build_type='Debug', required=False, timeout_s=120, commands=[
        ['cmake', '-E', 'make_directory', 'build/debug/isa', '../results/isa'],
        ['/opt/rocm/bin/hipcc', '--genco', '--offload-arch=gfx1151', '-g1', '-O2', '-std=c++17',
         '-DGGML_USE_HIP', '-Iexternal/ggml/include', '-Iexternal/ggml/src', '-Iexternal/ggml/src/ggml-cuda',
         'external/ggml/src/ggml-hip/gfx1151_v1/mmvq.cu', '-o', 'build/debug/isa/ar.bundle'],
        ['/opt/rocm/llvm/bin/clang-offload-bundler', '--unbundle', '--type=o',
         '--targets=hipv4-amdgcn-amd-amdhsa--gfx1151', '--input=build/debug/isa/ar.bundle',
         '--output=../results/isa/ar.hsaco']]))
    plan['suites'].append(dict(id='isa', build='isa', gpu=False, required=False, timeout_s=30,
        cases=[dict(id='disassembly', argv=['/opt/rocm/llvm/bin/llvm-objdump', '--disassemble', '--demangle', '../results/isa/ar.hsaco'])]))
    return plan


def ggml_v1_manifest(root):
    plan = operator_manifest(root)
    probe = ['build/debug/bin/test-backend-ops', 'test', '-b', 'ROCm0',
             '--suite', 'gfx1151_v1']
    groups = [('f32', ['-o', 'MUL_MAT', '-p', 'type_a=f32']),
              ('typed', ['-o', 'MUL_MAT', '-p', 'type_a=(f16|bf16)']),
              ('quant-fallback', ['-o', 'MUL_MAT', '-p', 'type_a=(q8_0|q4_0)']),
              ('copy', ['-o', 'CPY,CONT']), ('im2col', ['-o', 'IM2COL'])]
    checks = [dict(id=impl+'-'+name,
                   argv=probe+['--backend-impl', impl]+extra)
              for impl in ['default', 'gfx1151_v1'] for name, extra in groups]
    plan['suites'] = [
        dict(id='correctness', build='operators', gpu=True, timeout_s=180, cases=checks),
        dict(id='framework', build='device-health', gpu=True, timeout_s=15,
             cases=[dict(id='post-health', argv=['build/debug/device-health', 'health'])])]
    return plan


def ar_gemv_manifest(root):
    plan = operator_manifest(root)
    build = plan['builds'][1]
    build['commands'][0] += ['-DBUILD_SHARED_LIBS=OFF']
    build['commands'] += [
        ['cmake', '-E', 'copy', 'build/debug/bin/test-backend-ops', 'build/debug/bin/ar-rows1']]
    variants = [('stock', 1, 'default'), ('r1', 1, 'gfx1151_v1')]
    def argv(mode, rows, impl):
        return [f'build/debug/bin/ar-rows{rows}', mode, '-b', 'ROCm0',
                '--suite', 'gfx1151_v1', '--backend-impl', impl,
                '-o', 'MUL_MAT', '-p', r'type_a=q8_0,type_b=f32,m=[0-9]+,n=(1|2),']
    checks = [dict(id=name, argv=argv('test', rows, impl)) for name, rows, impl in variants]
    timed = []
    for repeat in range(2):
        for slot, (name, rows, impl) in enumerate(variants + list(reversed(variants))):
            timed.append(dict(id=f'r{repeat}-s{slot}-{name}', argv=argv('perf', rows, impl)))
    trace = ['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--output-format', 'csv',
             '--output-directory', '../results/profile/ar', '--'] + argv('test', 1, 'gfx1151_v1')
    plan['suites'] = [
        dict(id='correctness', build='operators', gpu=True, timeout_s=120, cases=checks),
        dict(id='framework', build='operators', gpu=True, timeout_s=180, cases=timed),
        dict(id='dispatch', build='operators', gpu=True, timeout_s=120,
             cases=[dict(id='q8-native-f32', argv=trace)])]
    return plan


def typed_manifest(root):
    plan = operator_manifest(root)
    probe = ['build/debug/bin/test-backend-ops']
    checks, timing = [], []
    for impl in ['default', 'halo']:
        for kind in ['f16', 'bf16']:
            checks.append(dict(id=impl+'-'+kind, argv=probe+[
                'test', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', impl,
                '-o', 'MUL_MAT', '-p', 'type_a='+kind]))
    for slot, impl in enumerate(['default', 'halo', 'halo', 'default']):
        for kind in ['f16', 'bf16']:
            timing.append(dict(id=f's{slot}-{impl}-{kind}', argv=probe+[
                'perf', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', impl,
                '-o', 'MUL_MAT', '-p', 'type_a='+kind]))
    timing.append(dict(id='trace', argv=['/opt/rocm/bin/rocprofv3', '--kernel-trace',
        '--output-format', 'csv', '--output-directory', '../results/profile/typed', '--']+probe+[
            'test', '-b', 'ROCm0', '--suite', 'halo', '--backend-impl', 'halo',
            '-o', 'MUL_MAT', '-p', 'type_a=(f16|bf16)']))
    plan['suites'] = [dict(id='correctness', build='operators', gpu=True, timeout_s=180, cases=checks),
                      dict(id='framework', build='operators', gpu=True, timeout_s=300, cases=timing)]
    return plan


def summarize_models(root, baseline_implementation='halo', families=None, candidate_implementation='halo', request_sets=None):
    report = {}
    for family, requests in (model_requests() if request_sets is None else request_sets).items():
        if families and family not in families:
            continue
        names = [r['id'] for r in requests]
        rows, audio = [], {}
        for variant in ['a1', 'b1', 'b2', 'a2']:
            label = family + '-' + variant
            text = (root / 'case' / label / 'session.log').read_text()
            expected = candidate_implementation if variant.startswith('b') else baseline_implementation
            command = json.loads(text.splitlines()[0])['argv']
            if 'session.wall_ms' not in text:
                raise ValueError('Missing canonical timing: ' + label)
            if expected == 'default':
                if '--backend-impl' in command or ('implementation=' + candidate_implementation) in text:
                    raise ValueError('Original HIP baseline must omit --backend-impl: ' + label)
            elif ('implementation=' + expected) not in text or '--backend-impl' not in command or command[command.index('--backend-impl') + 1] != expected:
                raise ValueError('Wrong implementation: ' + label)
            metrics = {}
            for name, key, value in re.findall(r'^metrics\[([^\]]+)\]\.([^=]+)=(.*)$', text, re.M):
                metrics.setdefault(name, {})[key] = value
            if set(metrics) != set(names):
                raise ValueError('Missing request metrics: ' + label)
            for name in names:
                file = root / 'wavs' / label / (name + '.wav')
                with wave.open(str(file), 'rb') as wav:
                    if wav.getsampwidth() != 2 or wav.getnchannels() not in (1, 2):
                        raise ValueError('Unexpected WAV format: ' + str(file))
                    pcm = array.array('h', wav.readframes(wav.getnframes()))
                    seconds = wav.getnframes() / wav.getframerate()
                    shape = (wav.getframerate(), wav.getnchannels())
                if not pcm or not any(pcm):
                    raise ValueError('Empty/silent WAV: ' + str(file))
                audio[variant, name] = (shape, pcm)
                wall = float(metrics[name]['wall_ms'])
                if not math.isfinite(wall) or wall <= 0 or abs(float(metrics[name]['audio_duration_ms']) - seconds * 1000) > 1:
                    raise ValueError('Invalid duration/timing: ' + label)
                if name != 'warmup':
                    rows.append(dict(variant='baseline' if variant.startswith('a') else 'candidate',
                                     session=variant, request=name, wall_ms=wall, audio_seconds=seconds,
                                     rtf=wall / (1000 * seconds)))
        aggregate = {}
        for variant in ['baseline', 'candidate']:
            group = [row for row in rows if row['variant'] == variant]
            wall, seconds = sum(r['wall_ms'] for r in group), sum(r['audio_seconds'] for r in group)
            aggregate[variant] = dict(wall_ms=wall, audio_seconds=seconds, rtf=wall / (1000 * seconds))
        parity = []
        for name in names[1:]:
            shape, reference = audio['a1', name]
            for variant in ['a2', 'b1', 'b2']:
                other_shape, candidate = audio[variant, name]
                row = dict(request=name, session=variant, equal_shape=shape == other_shape,
                           equal_length=len(reference) == len(candidate), pcm_identical=reference == candidate)
                if row['equal_shape'] and row['equal_length']:
                    aa, bb = sum(x*x for x in reference), sum(x*x for x in candidate)
                    error = sum((x-y)**2 for x,y in zip(reference, candidate))
                    row['cosine'] = sum(x*y for x,y in zip(reference, candidate)) / math.sqrt(aa*bb)
                    row['snr_db'] = 10 * math.log10(aa/error) if error else None
                parity.append(row)
        repeated, original = names[-1], names[1]
        repeats = {v: audio[v, original] == audio[v, repeated] for v in ['a1','b1','b2','a2']}
        report[family] = dict(rows=rows, aggregate=aggregate, parity=parity, repeat_pcm_identical=repeats)
    (root / 'comparison.json').write_text(json.dumps(dict(baseline_implementation=baseline_implementation, candidate_implementation=candidate_implementation),indent=2) + '\n')
    summary_path = root / 'models-summary.json'
    previous = json.loads(summary_path.read_text()) if request_sets is not None and summary_path.exists() else {}
    previous.update(report)
    summary_path.write_text(json.dumps(previous, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: dict(aggregate=v['aggregate'], repeat_pcm_identical=v['repeat_pcm_identical'])
                      for k,v in report.items()}, indent=2))
    return report


def summarize_operators(root, correctness_groups=10, timing_processes=24,
                        timing_glob='r*-s*-*.log', repeats=4):
    ansi = re.compile(r'\x1b\[[0-9;]*m')
    correctness = {}
    for path in sorted((root / 'case' / 'correctness').glob('*.log')):
        text = ansi.sub('', path.read_text())
        totals = re.findall(r'(\d+)/(\d+) tests passed', text)
        if not totals or any(int(a) != int(b) or int(b) == 0 for a, b in totals):
            raise ValueError('Incomplete or failed correctness group: ' + path.name)
        correctness[path.stem] = sum(int(a) for a, _ in totals)
    if len(correctness) != correctness_groups:
        raise ValueError('Missing correctness groups')
    measurements = {}
    pattern = re.compile(r'^\s+(\w+)\(([^\n]*)\):.*?(\d+) runs -\s+([0-9.]+) us/run', re.M | re.S)
    paths = sorted((root / 'case' / 'framework').glob(timing_glob))
    if len(paths) != timing_processes:
        raise ValueError('Missing timing processes')
    for path in paths:
        implementation = 'halo' if '-halo-' in path.name else 'default'
        matches = pattern.findall(ansi.sub('', path.read_text()))
        if not matches:
            raise ValueError('No timings in ' + path.name)
        for op, params, count, elapsed in matches:
            if int(count) <= 0 or float(elapsed) <= 0:
                raise ValueError('Invalid timing row in ' + path.name)
            measurements.setdefault((op, params), {}).setdefault(implementation, []).append(float(elapsed))
    rows = []
    for (op, params), variants in sorted(measurements.items()):
        if set(variants) != {'default', 'halo'} or any(len(v) != repeats for v in variants.values()):
            raise ValueError('Unpaired measurements: ' + params)
        a, b = (statistics.median(variants[k]) for k in ['default', 'halo'])
        rows.append(dict(op=op, params=params, default_us=a, halo_us=b,
                         speedup=a / b, default_min=min(variants['default']),
                         default_max=max(variants['default']), halo_min=min(variants['halo']),
                         halo_max=max(variants['halo'])))
    with (root / 'operators.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'operator-correctness.json').write_text(json.dumps(correctness, indent=2) + '\n')
    print(json.dumps(dict(correctness_checks=sum(correctness.values()), shapes=len(rows)), indent=2))
    return rows


def select_candidate(root, source):
    summary = json.loads((root / 'summary.json').read_text())
    if summary['status'] != 'PASS' or not summary['complete']:
        raise ValueError('Probes must finish successfully before any model runs')
    rows = summarize_operators(root, correctness_groups=6, timing_processes=16)
    kernels = set()
    for path in (root / 'profile/q8').rglob('*kernel_trace.csv'):
        with path.open() as stream:
            kernels.update(row['Kernel_Name'] for row in csv.DictReader(stream))
    grouped = [x for x in kernels if 'mul_mat_vec_q<(ggml_type)8,1,' in ''.join(x.split())
               and ',4>(' in ''.join(x.split())]
    if not grouped:
        raise ValueError('Trace did not execute the four-row Q8 kernel')
    actual = []
    for row in rows:
        if row['op'] != 'MUL_MAT':
            continue
        dims = {key: int(value) for key, value in re.findall(r'(?:^|,)(m|n|k)=(\d+)', row['params'])}
        if dims['n'] == 1:
            if dims['m'] not in [1024,3072]:
                raise ValueError('Unexpected output row count')
            actual.append((dims, row))
    if len(actual) != 10:
        raise ValueError('Expected ten actual one-token GEMV shapes')
    selected = None
    for threshold in [512,1024,2048]:
        group = [row for dims,row in actual if dims['k'] >= threshold]
        # Accept only a broad, repeatable regime, never one isolated fast shape.
        if len(group) >= 6 and all(float(r['default_min']) > 1.02 * float(r['halo_max']) for r in group) and \
                statistics.geometric_mean(float(r['speedup']) for r in group) >= 1.05:
            selected = threshold
            break
    candidate = '.'
    if selected:
        path = source / 'external/ggml/src/ggml-cuda/mmvq.cu'
        text = path.read_bytes()
        old = b'src0->ne[0] < 512 || src0->ne[1] < 1024'
        if text.count(old) != 1:
            raise ValueError('Q8 admission rule differs from the measured candidate')
        path.write_bytes(text.replace(old, ('src0->ne[0] < %d || src0->ne[1] < 1024' % selected).encode()))
    else:
        for name in ['external/ggml/src/ggml-cuda/mmvq.cu',
                     'external/ggml/src/ggml-cuda/ggml-cuda.cu',
                     'external/ggml/src/ggml-hip/implementation.cuh',
                     'external/ggml/src/ggml-hip/halo/halo.cuh']:
            shutil.copyfile(source.parent / 'source-copy' / name, source / name)
    if not any('halo_copy_rows' in name for name in kernels):
        raise ValueError('Trace did not execute the narrow-copy kernel')
    copy_min = 256
    for threshold in [64,128]:
        group = []
        for row in rows:
            if row['op'] != 'CPY':
                continue
            width, count = map(int,re.search(r'ne=\[(\d+),(\d+),',row['params']).groups())
            if width >= threshold and count > 1:
                group.append(row)
        if len(group) >= 4 and all(float(r['default_min']) > 1.02 * float(r['halo_max']) for r in group) and \
                statistics.geometric_mean(float(r['speedup']) for r in group) >= 1.05:
            copy_min = threshold
            break
    path = source / 'external/ggml/src/ggml-hip/halo/cpy.cu'
    if copy_min == 256:
        shutil.copyfile(source.parent / 'source-copy' / path.relative_to(source), path)
    else:
        text = path.read_text()
        if text.count('src->ne[0] < 64') != 1:
            raise ValueError('Narrow-copy admission rule differs from the probe')
        path.write_text(text.replace('src->ne[0] < 64','src->ne[0] < ' + str(copy_min)))
    kept = ['external/ggml/src/ggml-cuda/mmvq.cu', 'external/ggml/src/ggml-cuda/ggml-cuda.cu',
            'external/ggml/src/ggml-hip/implementation.cuh', 'external/ggml/src/ggml-hip/halo/halo.cuh',
            'external/ggml/src/ggml-hip/halo/cpy.cu']
    hashes = {}
    for name in kept:
        target = root / 'selected-source' / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    decision = dict(selected='copy-plus-q8' if selected else 'copy-only', min_k=selected,
        copy_min_columns=copy_min, selected_source_sha256=hashes,
        reason='Q8 accepted only with a stable >=5% aggregate win and separated ranges across >=6 shapes; otherwise rejected',
        grouped_q8_kernel_names=sorted(grouped), measurements=rows)
    (root / 'selection.json').write_text(json.dumps(decision,indent=2) + '\n')
    (source / 'model-campaign.json').write_text(json.dumps(model_manifest(source,candidate=candidate),indent=2) + '\n')
    print(json.dumps({k:v for k,v in decision.items() if k != 'measurements'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--baseline-implementation',choices=['default','halo','gfx1151_v1'],default='halo')
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--candidate-implementation', default='halo')
    parser.add_argument('--request-set', choices=['models', 'tts-main'], default='models')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--operators', action='store_true')
    actions.add_argument('--followup', action='store_true')
    actions.add_argument('--select-candidate', action='store_true')
    actions.add_argument('--models', action='store_true')
    actions.add_argument('--hip-baseline', action='store_true')
    actions.add_argument('--typed', action='store_true')
    actions.add_argument('--ggml-v1', action='store_true')
    actions.add_argument('--ar-gemv', action='store_true')
    actions.add_argument('--irodori-main', action='store_true')
    actions.add_argument('--tts-main', action='store_true')
    actions.add_argument('--tts-profile', action='store_true')
    actions.add_argument('--breeze-ar', action='store_true')
    actions.add_argument('--rdna-depth', action='store_true')
    actions.add_argument('--prepare-models', action='store_true')
    actions.add_argument('--summarize-operators', action='store_true')
    actions.add_argument('--summarize-models', action='store_true')
    args = parser.parse_args()
    if args.select_candidate:
        if args.source is None:
            parser.error('--select-candidate requires --source')
        select_candidate(args.root, args.source)
    elif args.prepare_models:
        prepare_models(args.root)
    elif args.summarize_operators:
        summarize_operators(args.root)
    elif args.summarize_models:
        summarize_models(args.root,args.baseline_implementation,args.families,args.candidate_implementation,
                         tts_requests() if args.request_set == 'tts-main' else None)
    else:
        plan = tts_profile_manifest(args.root) if args.tts_profile else rdna_depth_manifest(args.root) if args.rdna_depth else breeze_ar_manifest(args.root) if args.breeze_ar else ar_gemv_manifest(args.root) if args.ar_gemv else tts_main_manifest(args.root) if args.tts_main else irodori_main_manifest(args.root) if args.irodori_main else ggml_v1_manifest(args.root) if args.ggml_v1 else typed_manifest(args.root) if args.typed else hip_baseline_manifest(args.root) if args.hip_baseline else operator_manifest(args.root) if args.operators else followup_manifest(args.root) if args.followup else model_manifest(args.root) if args.models else manifest(args.root)
        (args.root / 'campaign.json').write_text(json.dumps(plan, indent=2) + '\n')
