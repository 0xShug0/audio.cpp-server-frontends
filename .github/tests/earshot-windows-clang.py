"""Run existing production targets/tests with the Windows release HIP compiler."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path.cwd()
logs = root / 'results'
logs.mkdir(exist_ok=True)
source = root / 'source'
build = source / 'build/debug'
env = os.environ.copy()

def run(args, name, expected=0):
    args = [str(x) for x in args]
    print('+', subprocess.list2cmdline(args), flush=True)
    with (logs / f'{name}.log').open('w', encoding='utf-8') as out:
        result = subprocess.run(args, env=env, stdout=out, stderr=subprocess.STDOUT)
    text = (logs / f'{name}.log').read_text(encoding='utf-8', errors='replace')
    print(text[-12000:], flush=True)
    if expected is not None and result.returncode != expected:
        raise RuntimeError(f'{name} exited {result.returncode}')
    return result.returncode, text

run([sys.executable, '-m', 'pip', 'install', 'ninja', '--index-url', 'https://pypi.org/simple'], 'install-ninja')
run([sys.executable, '-m', 'pip', 'install', '--index-url', 'https://stable.repo.amd.com/rocm/whl-next/', 'rocm[libraries,devel]==10.0.0'], 'install-rocm')
run(['rocm-sdk', 'init'], 'init-rocm')
rocm = Path(subprocess.check_output(['rocm-sdk', 'path', '--root'], text=True).strip())
rocm_bin = subprocess.check_output(['rocm-sdk', 'path', '--bin'], text=True).strip()
rocm_cmake = subprocess.check_output(['rocm-sdk', 'path', '--cmake'], text=True).strip()
llvm = rocm / 'lib/llvm/bin'
env.update(ROCM_PATH=str(rocm), HIP_PATH=str(rocm), HIP_PLATFORM='amd',
           CMAKE_PREFIX_PATH=rocm_cmake, PATH=rocm_bin + os.pathsep + env['PATH'],
           HIP_DEVICE_LIB_PATH=str(rocm / 'lib/llvm/amdgcn/bitcode'), LLVM_PATH=str(rocm / 'lib/llvm'))
run([llvm / 'clang++.exe', '--version'], 'compiler')
run(['pwsh', '-NoProfile', '-File', source / 'scripts/build_windows_hip.ps1',
     '-RocmPath', rocm, '-GpuTargets', 'gfx1151', '-BuildDir', build,
     '-BuildType', 'Debug', '-NoNativeCpu', '-DeploymentBuild', '-Graphs', '-ConfigureOnly'], 'configure-hip')
args = ['cmake', '-S', source, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Debug',
        '-DCMAKE_CXX_STANDARD=17', '-DENGINE_ENABLE_HIP=ON', '-DENGINE_ENABLE_CUDA=OFF',
        '-DENGINE_ENABLE_VULKAN=OFF', '-DENGINE_ENABLE_NATIVE_CPU=OFF',
        '-DENGINE_ENABLE_CPU_ALL_VARIANTS=ON', '-DENGINE_ENABLE_OPENMP=OFF', '-DGGML_OPENMP=OFF',
        '-DAUDIOCPP_DEPLOYMENT_BUILD=ON', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF',
        '-DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF', '-DAUDIOCPP_MODEL_SET=custom',
        '-DAUDIOCPP_MODELS=earshot_vad', '-DENGINE_BUILD_TESTS=ON',
        '-DENGINE_BUILD_MODEL_TESTS=OFF', '-DAUDIOCPP_BUILD_C_API=OFF',
        '-DENGINE_ENABLE_CUDA_GRAPHS=ON', '-DGPU_TARGETS=gfx1151',
        f'-DCMAKE_PREFIX_PATH={rocm_cmake}', f'-DCMAKE_C_COMPILER={llvm / "clang.exe"}',
        f'-DCMAKE_CXX_COMPILER={llvm / "clang++.exe"}',
        '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types',
        f'-DCMAKE_CXX_FLAGS=-isystem "{(source / "external/cpp-httplib").as_posix()}"',
        '-DGGML_HIP_NO_VMM=ON']
if shutil.which('ccache'):
    args += ['-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache']
run(args, 'configure')
command = ['cmake', '--build', build, '--parallel', '8', '--target', 'audiocpp_cli', 'audiocpp_server', 'earshot_vad_test']
code, text = run(command, 'before-build', expected=None)
if code == 0 or 'undefined symbol: __cpu_model' not in text:
    raise RuntimeError('Expected original __cpu_model link failure was not reproduced')
run(['git', '-C', source, 'apply', '--check', root / '.github/tests/earshot-windows-clang.patch'], 'patch-check')
run(['git', '-C', source, 'apply', root / '.github/tests/earshot-windows-clang.patch'], 'patch-apply')
run(command, 'after-build')
run([build / 'bin/earshot_vad_test.exe', '--log'], 'earshot-avx2')
env['AUDIOCPP_EARSHOT_BASELINE'] = '1'
run([build / 'bin/earshot_vad_test.exe', '--log'], 'earshot-sse2')
del env['AUDIOCPP_EARSHOT_BASELINE']
run([build / 'bin/audiocpp_cli.exe', '--help', '--log'], 'cli-help')
run([build / 'bin/audiocpp_server.exe', '--help', '--log'], 'server-help')
shutil.copy2(build / 'CMakeCache.txt', logs / 'CMakeCache.txt')
summary = 'Original Windows ROCm Clang __cpu_model link failure reproduced. Exact two-guard fix: Debug C++17 HIP CLI/server link and start; existing Earshot parity, streaming, loader and VAD tests pass with AVX2 and forced SSE2. No release published.'
(logs / 'summary.txt').write_text(summary)
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as out:
    out.write(summary + '\n')
