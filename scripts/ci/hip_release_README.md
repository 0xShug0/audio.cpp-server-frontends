# audio.cpp HIP/ROCm prebuilt

Requires an AMD GPU supported by ROCm @ROCM_VERSION@, its compatible driver,
and the matching ROCm runtime libraries. Linux requires Ubuntu 24.04 or newer
and an AVX2/FMA/F16C-capable x64 CPU. Windows selects a CPU implementation at runtime.
`BUILD.json` lists the compiled GPU architectures and exact source revision.
Models are downloaded separately from audio-cpp on Hugging Face.

Install the matching runtime below, or use an existing ROCm @ROCM_VERSION@
installation. The compiler/devel package is only needed to build. Python with
venv/pip support is needed for the runtime installation commands.

Linux, after extracting the binary archive (Ubuntu's `python3-venv` package
provides venv support):

```sh
python3 -m venv rocm-runtime
rocm-runtime/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries]==@ROCM_VERSION@'
export LD_LIBRARY_PATH="$(rocm-runtime/bin/python -c 'import os,rocm_sdk; print(os.pathsep.join(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries("amdhip64","hipblas","hipblaslt"))))'):${LD_LIBRARY_PATH:-}"
./audiocpp_cli --list-devices
./audiocpp_cli --task asr --family moonshine_asr --backend hip --model /path/to/model.gguf --audio /path/to/speech.wav --log --metrics
./audiocpp_server --ui --ui-management --backend hip
```

Windows PowerShell, after extracting the binary ZIP:

```powershell
py -3 -m venv rocm-runtime
.\rocm-runtime\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries]==@ROCM_VERSION@'
$rocm = & .\rocm-runtime\Scripts\python.exe -c 'import os,rocm_sdk; print(os.pathsep.join(dict.fromkeys(str(p.parent) for p in rocm_sdk.find_libraries("amdhip64","hipblas","hipblaslt"))))'
$env:PATH = "$rocm;$env:PATH"
.\audiocpp_cli.exe --list-devices
.\audiocpp_cli.exe --task asr --family moonshine_asr --backend hip --model C:\path\to\model.gguf --audio C:\path\to\speech.wav --log --metrics
.\audiocpp_server.exe --ui --ui-management --backend hip
```

Open the server UI at http://127.0.0.1:8080. Keep the shared libraries beside
the executables. The library archive has `include/audiocpp.h` and `libs/` for
embedding. Its libraries must stay together; configure the application's
library search path to include `libs/` and the ROCm runtime. For Windows
applications, place the contents of `libs/` beside the application executable
so the GGML backend plugins are discovered.

The native model manager and UI are included. Models using eSpeak additionally
need eSpeak-ng and its data installed. No models or Python inference engine are
bundled. `SHA256SUMS` verifies the archive downloads.
