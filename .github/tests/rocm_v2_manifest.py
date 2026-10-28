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
builds.append(dict(id='framework', build_type='Debug', depends_on=['ggml'], timeout_s=240,
    commands=[['cmake', '--build', 'build/debug', '--target', 'framework_backend_probe',
               'backend_config_args_test', '--parallel', '8']]))
vulkan_setup = """set -euo pipefail
mkdir -p toolchain/debs toolchain/vulkan
cd toolchain/debs
apt-get download libvulkan-dev glslc libshaderc1 spirv-headers
for package in *.deb; do dpkg-deb -x "$package" ../vulkan; done
"""
vulkan_configure = """set -euo pipefail
export LD_LIBRARY_PATH="$PWD/toolchain/vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  '-DCMAKE_CXX_FLAGS_DEBUG=-g1 -O2' '-DCMAKE_C_FLAGS_DEBUG=-g1 -O2' \
  -DCMAKE_PREFIX_PATH="$PWD/toolchain/vulkan/usr" \
  -DVulkan_LIBRARY=/lib/x86_64-linux-gnu/libvulkan.so.1 \
  -DVulkan_INCLUDE_DIR="$PWD/toolchain/vulkan/usr/include" \
  -DVulkan_GLSLC_EXECUTABLE="$PWD/toolchain/vulkan/usr/bin/glslc" \
  -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=ON -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF
cmake --build build/debug --target test-backend-ops --parallel 8
"""
builds.append(dict(id='vulkan', build_type='Debug', cwd='../source-vulkan', timeout_s=480,
    commands=[['bash', '-c', vulkan_setup], ['bash', '-c', vulkan_configure]]))
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
# Check tiny-case timing against two unchanged HIP instances and reversed order.
control_indices = [14,17,22,25,51,55,61,65,84,90,95,105,111,112]
suites.append(dict(id='controls', build='ggml', gpu=True, timeout_s=90, cases=[
    dict(id=f'{index:03d}-{label}', argv=probe + ['perf'] + selection +
         ['--case-index',str(index),'--backend-params','', '--compare-params',comparison])
    for index in control_indices for label,comparison in [('same-hip',''),('reversed','rocm_v2')]]))
suites.append(dict(id='framework', build='framework', gpu=True, timeout_s=45, cases=[
    dict(id='selection', argv=['build/debug/bin/framework_backend_probe', '--require-rocm-v2', '--log']),
    dict(id='parsing', argv=['build/debug/bin/backend_config_args_test', '--log'])]))
# Same deterministic input producer and cases as the HIP campaign. Each worker
# compares two instances of Vulkan to give a warmed wall-time reference.
vulkan_indices = list(range(8)) + [21,22,24,25,35,36,41,44,46,48,50,64,66,68,70,72,74,124,125] + list(range(127,142))
suites.append(dict(id='vulkan', build='vulkan', gpu=True, timeout_s=30, cases=[
    dict(id=f'{index:03d}-reference', argv=['../source-vulkan/build/debug/bin/test-backend-ops', 'perf',
         '--suite','rocm-v2','-b','Vulkan0','--case-index',str(index),'--compare-params',''])
    for index in vulkan_indices]))
suites.append(dict(id='vulkan-correctness', build='vulkan', gpu=True, timeout_s=60, cases=[
    dict(id=f'{index:03d}-cpu-reference', argv=['../source-vulkan/build/debug/bin/test-backend-ops', 'test',
         '--suite','rocm-v2','-b','Vulkan0','--case-index',str(index)]) for index in (0,1,125)]))
# This independently compiled health worker does not depend on the integrated backend.
health = dict(argv=['build/debug/device-health', 'health'], timeout_s=15)
manifest = dict(version=1, builds=builds, suites=suites, health_check=health)
(root / 'campaign.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Campaign: {len(builds)} independent builds, {len(correctness)} correctness cases, '
      f'{len(paired)} paired graphs, {sum(len(s["cases"]) for s in suites[2:])} independent candidate workers')
