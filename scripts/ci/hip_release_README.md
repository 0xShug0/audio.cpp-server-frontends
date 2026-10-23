# audio.cpp HIP/ROCm prebuilt

Requires an AMD GPU supported by ROCm @ROCM_VERSION@, its compatible driver,
and the matching ROCm runtime libraries. Linux requires Ubuntu 24.04 or newer.
`BUILD.json` lists the compiled GPU architectures and exact source revision.
Models are downloaded separately from audio-cpp on Hugging Face.

Install the matching runtime (the compiler/devel package is only needed to build):

```sh
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries]==@ROCM_VERSION@'
```

Linux, after extracting the binary archive:

```sh
export LD_LIBRARY_PATH="$(rocm-sdk path --root)/lib:${LD_LIBRARY_PATH:-}"
./audiocpp_cli --list-devices
./audiocpp_cli --task asr --family moonshine_asr --backend hip --model /path/to/model.gguf --audio /path/to/speech.wav --log --metrics
./audiocpp_server --ui --ui-management --backend hip
```

Windows PowerShell, after extracting the binary ZIP:

```powershell
$env:PATH = "$(rocm-sdk path --bin);$env:PATH"
.\audiocpp_cli.exe --list-devices
.\audiocpp_cli.exe --task asr --family moonshine_asr --backend hip --model C:\path\to\model.gguf --audio C:\path\to\speech.wav --log --metrics
.\audiocpp_server.exe --ui --ui-management --backend hip
```

Open the server UI at http://127.0.0.1:8080. Keep the shared libraries beside
the executables. The library archive has `include/audiocpp.h` and `libs/` for
embedding. Its libraries must stay together; configure the application's
library search path to include `libs/` and the ROCm runtime.

The native model manager and UI are included. Models using eSpeak additionally
need eSpeak-ng and its data installed. No models or Python inference engine are
bundled. `SHA256SUMS` verifies the archive downloads.
