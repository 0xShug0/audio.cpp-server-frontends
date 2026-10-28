"""Construct the fixed ROCm-v2 CI campaign from the locally checked GGML case list."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root
entries = []
for line in (root / '.github/tests/rocm-v2-cases.tsv').read_text().splitlines():
    number, op, params = line.split('\t', 2)
    if int(number) != len(entries):
        raise ValueError('case indices must be contiguous and ordered')
    entries.append((int(number), op, params))
if not entries:
    raise ValueError('empty GGML campaign')
cmake = ['cmake', '-S', 'external/ggml', '-B', 'build/debug', '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Debug', '-DCMAKE_CXX_STANDARD=17',
         '-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang',
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
         '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
         '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
         '-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++',
         '-DCMAKE_HIP_FLAGS_DEBUG=-g1 -O2', '-DCMAKE_CXX_FLAGS_DEBUG=-g1 -O2',
         '-DCMAKE_C_FLAGS_DEBUG=-g1 -O2', '-DCMAKE_PREFIX_PATH=/opt/rocm',
         '-DGPU_TARGETS=gfx1151', '-DGGML_HIP=ON', '-DGGML_CUDA=OFF',
         '-DGGML_VULKAN=OFF', '-DGGML_OPENMP=ON', '-DGGML_HIP_NO_VMM=ON',
         '-DGGML_BUILD_TESTS=ON', '-DGGML_BUILD_EXAMPLES=OFF']
hipcc = ['/opt/rocm/bin/hipcc', '--offload-arch=gfx1151', '-g', '-O2', '-std=c++17']
builds = [
    dict(id='device-health', build_type='Debug', timeout_s=120, commands=[hipcc + [
         '.github/tests/hip_vmm_probe.cpp', '-o', 'build/debug/device-health']]),
    dict(id='ggml', build_type='Debug', timeout_s=1200, commands=[cmake,
         ['cmake', '--build', 'build/debug', '--target', 'test-backend-ops', '--parallel', '8']]),
    dict(id='sgemm-candidate', build_type='Debug', timeout_s=180, commands=[hipcc + [
         '-Iexternal/ggml/src/ggml-hip/halo', 'tests/unittests/test_hip_halo_sgemm.cu',
         'external/ggml/src/ggml-hip/halo/sgemm.cu', '-lhipblaslt', '-o', 'build/debug/raw-sgemm']]),
    dict(id='mmvq-candidates', build_type='Debug', timeout_s=240, commands=[
         ['python3', '.github/tests/extract_halo_mmvq_probe.py'], hipcc + [
         '-DGGML_USE_HIP', '-I.', '-Iexternal/ggml/include', '-Iexternal/ggml/src',
         '-Iexternal/ggml/src/ggml-cuda', 'tests/unittests/test_hip_halo_mmvq.cu',
         'external/ggml/src/ggml-hip/halo/mmvq.cu', '-o', 'build/debug/raw-mmvq']]),
    dict(id='im2col-candidate', build_type='Debug', timeout_s=180, commands=[hipcc + [
         '-Iexternal/ggml/include', '-Iexternal/ggml/src/ggml-hip/halo',
         'tests/unittests/test_hip_halo_im2col.cu', 'external/ggml/src/ggml-hip/halo/im2col.cu',
         '-o', 'build/debug/raw-im2col']]),
]
# Raw targets remain independently executable if the integrated backend fails to build.
for build in builds:
    build['commands'].insert(0, ['cmake', '-E', 'make_directory', 'build/debug'])
probe = ['build/debug/bin/test-backend-ops']
selection = ['--suite', 'rocm-v2', '-b', 'ROCm0']
correctness, paired = [], []
for index, op, params in entries:
    name = f'{index:03d}-{op.lower()}'
    case = ['--case-index', str(index)]
    if 'expected_status=GGML_STATUS_FAILED' in params:
        correctness.append(dict(id=name, argv=probe + ['test'] + selection + case +
                                ['--backend-params', 'rocm_v2']))
        continue
    # CPU GGML has no BF16 IM2COL implementation; the paired path checks it against
    # inherited HIP, and the independent raw suite checks scalar rounding and guards.
    if not (op == 'IM2COL' and 'dst_type=bf16' in params):
        correctness.append(dict(id=name, argv=probe + ['test'] + selection + case +
                                ['--backend-params', 'rocm_v2']))
    paired.append(dict(id=name, argv=probe + ['perf'] + selection + case +
                       ['--backend-params', 'rocm_v2', '--compare-params', '']))
sgemm_shapes = {(m, n, k, pad) for pad in (0, 3) for m, n, k in (
    (1,1,1), (31,33,65), (64,64,64), (67,131,97), (512,128,1536),
    (256,1024,3584), (128,8192,1792), (64,32768,896), (1024,128,1024), (128,550,3200))}
for line in (root / 'tests/unittests/assets/hip_halo/qwen3_sgemm_shapes.txt').read_text().splitlines():
    if line.startswith('[GEMM_SHAPE] '):
        m,n,k,ta,tb,tc,lda,ldb,ldc,batches = map(int, line.split()[1:])
        if ta == tb == tc == 0 and batches == 1:
            sgemm_shapes.add((m,n,k,0))
mmvq_shapes = [(k,m,gate,0) for gate in (0,1) for k,m in (
    (32,1), (96,3), (512,67), (512,512), (512,1024), (512,2048),
    (1024,512), (1024,1024), (1024,2048), (1024,3072), (1024,4096),
    (2048,512), (2048,1024), (2048,2048), (2048,3072), (2048,6144),
    (3072,1024), (4096,1024), (6144,2048))] + [(512,65,gate,3) for gate in (0,1)]
suites = [
    dict(id='correctness', build='ggml', gpu=True, timeout_s=90, cases=correctness),
    dict(id='paired', build='ggml', gpu=True, timeout_s=90, cases=paired),
    dict(id='independent-sgemm', build='sgemm-candidate', gpu=True, timeout_s=45, cases=[
         dict(id='shape-' + '-'.join(map(str, shape)),
              argv=['build/debug/raw-sgemm'] + list(map(str, shape))) for shape in sorted(sgemm_shapes)]),
    dict(id='independent-mmvq', build='mmvq-candidates', gpu=True, timeout_s=45, cases=[
         dict(id='shape-' + '-'.join(map(str, shape)),
              argv=['build/debug/raw-mmvq'] + list(map(str, shape))) for shape in mmvq_shapes]),
    dict(id='independent-im2col', build='im2col-candidate', gpu=True, timeout_s=60, cases=[
         dict(id=f'cases-{start}-{min(start+31,458)}',
              argv=['build/debug/raw-im2col', '--range', str(start), '32']) for start in range(0,459,32)]),
]
# This independently compiled health worker does not depend on the integrated backend.
health = dict(argv=['build/debug/device-health', 'health'], timeout_s=15)
manifest = dict(version=1, builds=builds, suites=suites, health_check=health)
(root / 'campaign.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Campaign: {len(builds)} independent builds, {len(correctness)} correctness cases, '
      f'{len(paired)} paired graphs, {sum(len(s["cases"]) for s in suites[2:])} independent candidate workers')
