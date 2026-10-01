# audio_decode

`audio_decode` is a pre-processing module for ASR requests:

```text
POST /v1/audio/transcriptions
client_encoded_audio_request -> core_wav_audio_request
```

It accepts MP3 and FLAC input through JSON paths or multipart uploads, decodes
the file with miniaudio, writes a temporary WAV, rewrites the request to a
core-compatible JSON request, then lets the core ASR handler run normally.

## Build

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  -DAUDIOCPP_SERVER_FRONTENDS_DIR=/path/to/audio.cpp-server-frontends \
  -DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode
```

`audio_decode` uses the vendored miniaudio header and does not add a system
library dependency.

## Open WebUI

Use the standard (full) Open WebUI image, such as
`ghcr.io/open-webui/open-webui:main`, for microphone transcription. Its slim
images omit FFmpeg and bypass audio preprocessing, so browser recordings can
reach audio.cpp as WebM and receive HTTP 400. This frontend accepts MP3/FLAC in
addition to the core's WAV input; it does not decode WebM/Opus.

The full Open WebUI image can transcode these recordings to MP3 before sending
them. Leave `BYPASS_PYDUB_PREPROCESSING` unset or `false` in Open WebUI. FFmpeg
in the audio.cpp container does not perform this client-side conversion.

## Request

Multipart upload:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=qwen3-asr \
  -F file=@speech.mp3
```

JSON path:

```json
{
  "model": "qwen3-asr",
  "file": "speech.flac"
}
```
