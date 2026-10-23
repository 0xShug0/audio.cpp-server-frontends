#!/usr/bin/env python3
"""Build full HIP deployments; emit release archives, never publish them."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    windows = platform.system() == 'Windows'
    version = os.environ['RELEASE_VERSION']
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_' for c in version):
        raise ValueError('version must be an archive-safe version string')
    rocm_version = os.environ['ROCM_VERSION']
    targets = os.environ['GPU_TARGETS']
    if not targets or any(not arch.startswith('gfx') or not arch.isalnum() for arch in targets.split(';')):
        raise ValueError('GPU_TARGETS must be semicolon-separated gfx architecture names')
    env = os.environ.copy()

    def run(command, name, cwd=None):
        print('+', subprocess.list2cmdline([str(x) for x in command]), flush=True)
        with (logs / (name + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            if process.wait():
                raise RuntimeError(f'{name} failed: {process.returncode}; see {log.name}')

    run([sys.executable, '-m', 'pip', 'install', '--index-url',
         'https://stable.repo.amd.com/rocm/whl-next/',
         f'rocm[libraries,devel]=={rocm_version}'], 'install-rocm')
    run(['rocm-sdk', 'init'], 'init-rocm')
    rocm = Path(subprocess.check_output(['rocm-sdk', 'path', '--root'], text=True).strip())
    rocm_bin = subprocess.check_output(['rocm-sdk', 'path', '--bin'], text=True).strip()
    rocm_cmake = subprocess.check_output(['rocm-sdk', 'path', '--cmake'], text=True).strip()
    env.update(ROCM_PATH=str(rocm), HIP_PATH=str(rocm), HIP_PLATFORM='amd',
               CMAKE_PREFIX_PATH=rocm_cmake, PATH=rocm_bin + os.pathsep + env['PATH'])
    if not windows:
        env['LD_LIBRARY_PATH'] = str(rocm / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    build = output / 'build'
    command = ['cmake', '-S', source, '-B', build, '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DENGINE_ENABLE_HIP=ON',
               '-DENGINE_ENABLE_CUDA=OFF', '-DENGINE_ENABLE_VULKAN=OFF',
               '-DENGINE_ENABLE_NATIVE_CPU=OFF', '-DENGINE_ENABLE_CPU_ALL_VARIANTS=ON',
               '-DENGINE_ENABLE_OPENMP=OFF', '-DGGML_OPENMP=OFF',
               '-DAUDIOCPP_DEPLOYMENT_BUILD=ON', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON',
               '-DAUDIOCPP_BUILD_C_API=ON', '-DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF',
               '-DENGINE_ENABLE_CUDA_GRAPHS=ON', f'-DAUDIOCPP_VERSION={version}',
               f'-DGPU_TARGETS={targets}', f'-DCMAKE_PREFIX_PATH={rocm_cmake}']
    if windows:
        llvm = rocm / 'lib' / 'llvm' / 'bin'
        command += [f'-DCMAKE_C_COMPILER={llvm / "clang.exe"}',
                    f'-DCMAKE_CXX_COMPILER={llvm / "clang++.exe"}',
                    '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types',
                    '-DGGML_HIP_NO_VMM=ON']
    else:
        llvm = Path(subprocess.check_output([str(Path(rocm_bin) / 'hipconfig'), '-l'], env=env, text=True).strip())
        command += [f'-DCMAKE_HIP_COMPILER={llvm / "clang"}',
                    '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON']
    run(command, 'configure')
    run(['cmake', '--build', build, '--parallel', str(min(os.cpu_count() or 2, 8)),
         '--target', 'audiocpp_cli', 'audiocpp_server', 'audiocpp_gguf', 'audiocpp'], 'build')
    shutil.copy2(build / 'CMakeCache.txt', logs / 'CMakeCache.txt')
    binary_dir = build / 'bin'
    if not list(binary_dir.glob('ggml-hip*.dll' if windows else 'libggml-hip.so*')):
        raise RuntimeError('HIP backend library was not produced')
    suffix = 'windows-x64' if windows else 'ubuntu-x64'
    label = f'audio-{version}-{{kind}}-{suffix}-rocm{rocm_version}'
    manifest = dict(source_repository=os.environ.get('SOURCE_REPOSITORY', '0xShug0/audio.cpp'),
                    source_sha=subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip(),
                    rocm_version=rocm_version, gpu_targets=targets.split(';'),
                    cpu='portable runtime dispatch', deployment=True, full_model_build=True)
    archives = []
    for kind in ('bin', 'lib'):
        stage = output / label.format(kind=kind)
        stage.mkdir()
        destination = stage if kind == 'bin' else stage / 'libs'
        destination.mkdir(exist_ok=True)
        for path in binary_dir.iterdir():
            is_library = path.suffix in ('.dll', '.lib') if windows else '.so' in path.name
            if path.is_file() and (is_library or kind == 'bin' and path.name in ('audiocpp_cli', 'audiocpp_server', 'audiocpp_gguf', 'audiocpp_cli.exe', 'audiocpp_server.exe', 'audiocpp_gguf.exe')):
                shutil.copy2(path, destination / path.name, follow_symlinks=False)
        if windows:
            # Match the HIP DLL to the build toolchain, not a driver-installed DLL.
            for name in ('amdhip64_7.dll', 'rocm_kpack.dll', 'amd_comgr.dll'):
                shutil.copy2(Path(rocm_bin) / name, destination / name)
            vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
            vs = Path(subprocess.check_output([vswhere, '-latest', '-property', 'installationPath'], text=True).strip())
            redist = sorted((vs / 'VC/Redist/MSVC').glob('*/x64/Microsoft.VC143.CRT'))[-1]
            for pattern in ('vcruntime140*.dll', 'msvcp140*.dll'):
                for dll in redist.glob(pattern):
                    shutil.copy2(dll, destination / dll.name)
        else:
            for library in destination.glob('*.so*'):
                run(['patchelf', '--set-rpath', '$ORIGIN', library], 'rpath-' + kind + '-' + library.name)
        if kind == 'lib':
            (stage / 'include').mkdir()
            shutil.copy2(source / 'include/audiocpp.h', stage / 'include')
        else:
            shutil.copytree(source / 'model_specs', stage / 'model_specs')
            shutil.copytree(source / 'tools', stage / 'tools', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(source / 'LICENSE', stage)
        (stage / 'BUILD.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        readme = Path(__file__).with_name('hip_release_README.md').read_text(encoding='utf-8')
        (stage / 'README.md').write_text(readme.replace('@ROCM_VERSION@', rocm_version), encoding='utf-8')
        if windows:
            archive = output / (stage.name + '.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as pack:
                for path in sorted(stage.rglob('*')):
                    if path.is_file():
                        pack.write(path, path.relative_to(stage))
        else:
            archive = output / (stage.name + '.tar.gz')
            with tarfile.open(archive, 'w:gz') as pack:
                for path in sorted(stage.iterdir()):
                    pack.add(path, arcname=path.name)
        archives.append(archive)
    with (output / 'SHA256SUMS').open('w', encoding='utf-8') as sums:
        for path in archives:
            with path.open('rb') as file:
                digest = hashlib.file_digest(file, 'sha256').hexdigest()
            sums.write(f'{digest}  {path.name}\n')


if __name__ == '__main__':
    main()
