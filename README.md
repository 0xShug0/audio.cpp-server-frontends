# Server Frontends

Server frontends are optional in-process adapters around the stable server core.
The core route handlers keep their normal WAV/text request and response
contract. Frontends adapt client-facing behavior without adding format- or
client-specific code to the core server.

Frontend support is off by default in the main audio.cpp build:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
```

Enable frontends and select the modules/capabilities explicitly:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  -DAUDIOCPP_SERVER_FRONTENDS_DIR=/path/to/audio.cpp-server-frontends \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;mp3_encode'
```

`AUDIOCPP_SERVER_FRONTENDS_DIR` points at this package. It may be a sibling
checkout, an optional git submodule such as `external/audio.cpp-server-frontends`,
or any other local path. `AUDIOCPP_SERVER_FRONTEND_MODULES` is a semicolon-separated
ordered list. Only the selected sources and their private dependencies are
compiled and linked into `audiocpp_server`.

## Docker images

The **Docker (all frontends)** workflow builds CPU, Vulkan, CUDA 12.9.2, and CUDA 13.0.2 images
for Linux amd64 and arm64. Use CUDA 13 for DGX Spark. All four modules below are compiled in. Images
use a Debug build and include all model families; no model weights are included.
Plain HTTP remains the default; HTTPS and WebSocket listeners require explicit
runtime selection.

Daily at 15:21 UTC, the workflow checks for the latest stable audio.cpp release
and automatically builds and publishes it with the frontend default branch,
updating `:cpu`, `:vulkan`, `:cuda12`, and `:cuda13`. This tracks releases, not every audio.cpp commit. The workflow
caches successful publications by both source revisions and skips unchanged
pairs; failed builds are retried on the next check. Cache eviction can cause a
rebuild. This is twelve hours after audio.cpp's 03:21 UTC Docker schedule;
the daily build is skipped if an audio.cpp Docker run is still active.
Scheduled runs require this workflow on the default branch.

Run the workflow manually to test a pinned audio.cpp tag or commit. Publishing
is off by default for manual runs. Publishing a frontend GitHub release builds
the audio.cpp revision pinned in the workflow and publishes to
`ghcr.io/0xshug0/audio.cpp-server-frontends:<release-tag>-<backend>`, where
`<backend>` is `cpu`, `vulkan`, `cuda12`, or `cuda13`. Stable releases also update
the corresponding backend tags; manual published builds use a unique run-based
tag for each backend.
The image labels and workflow summary record both source revisions.

Once an image has been published:

CPU (no GPU or NVIDIA container runtime required):

```bash
docker run --rm -p 8080:8080 \
  -v /path/to/models:/app/models \
  ghcr.io/0xshug0/audio.cpp-server-frontends:cpu
```

Vulkan on Linux with AMD/Intel graphics:

```bash
mkdir -p ./vulkan-cache

docker run --rm -p 8080:8080 \
  --user "$(id -u):$(id -g)" \
  --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -e XDG_CACHE_HOME=/cache \
  -v "$PWD/vulkan-cache:/cache" \
  -v /path/to/models:/app/models \
  ghcr.io/0xshug0/audio.cpp-server-frontends:vulkan
```

Use your GPU's render node if it is not `renderD128`. The image includes the
Vulkan loader and Mesa drivers; the host must provide a supported GPU and kernel
driver. The supplementary group grants the container's non-root user access to
the render node. NVIDIA Vulkan requires the NVIDIA container runtime and its
graphics driver libraries instead of Mesa's AMD/Intel drivers.

The cache mount preserves Mesa's shader cache when the container is recreated.
`--user` matches the host user's ownership of `vulkan-cache`; if you use another
UID/GID or an existing volume, ensure that user can write to it. Setting
`XDG_CACHE_HOME` explicitly also avoids attempts to write to `//.cache` when a
custom container user has no home directory. An unwritable cache disables Mesa's
disk cache and produces a warning.

CUDA:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v /path/to/models:/app/models \
  ghcr.io/0xshug0/audio.cpp-server-frontends:cuda13
```

Use `:cuda12` instead for the CUDA 12 image.

The default command opens the full WebUI. To use a server configuration, mount
it and pass `--config /path/in/container/server.json --host 0.0.0.0 --port 8080
--log` after the image name. The mounted model directory must be writable by
the container user (`ubuntu` by default) when downloading through the UI.

The build runners do not have GPUs. Publishing checks build/link dependencies,
not GPU inference; MP3 and GPU end-to-end testing still requires a running
server and model weights.

For Open WebUI microphone transcription, use its standard (full) image rather
than a slim image so recordings can be transcoded before upload. See
[Open WebUI input formats](docs/audio_decode.md#open-webui).

## Supported Frontends

| Name | Type | Purpose | Docs |
|---|---|---|---|
| `audio_decode` | pre-processing module | Accept MP3/FLAC ASR input and rewrite it to core WAV input | [audio_decode.md](docs/audio_decode.md) |
| `mp3_encode` | pre/post-processing module | Honor TTS `response_format=mp3` by encoding core WAV output to MP3 | [mp3_encode.md](docs/mp3_encode.md) |
| `https` | listener capability | Serve the same in-process server over HTTPS | [https.md](docs/https.md) |
| `websocket` | listener capability | Serve HTTP plus a WebSocket bridge to the same in-process handler | [websocket.md](docs/websocket.md) |

For adding a new module, see [adding_modules.md](docs/adding_modules.md).

## Pipeline

For each HTTP request, the server runs:

```text
client request
  -> selected module pre_process(), in configured order
  -> core server request handler
  -> selected module post_process(), in configured order
  -> client response
```

A pre-processing module mutates `ServerFrontendRequest::request`. It can also
set `ServerFrontendRequest::response` to stop before the core handler and return
an explicit error or replacement response.

A post-processing module mutates `ServerFrontendResponse::response` after the
core handler returns. It also receives both the original client request and the
request that actually reached core.

Listener capabilities such as `https` and `websocket` are not part of this
pre/post pipeline. They are frontend-owned transports selected by name through
audio.cpp's generic `frontend_listener` connector, then forward requests to the
same core handler.

## Contracts

Each active module side declares the HTTP envelope state it accepts and emits:

```cpp
struct FrontendPreContract {
    std::string_view method;
    std::string_view path;
    std::string_view request_in;
    std::string_view request_out;
};

struct FrontendPostContract {
    std::string_view method;
    std::string_view path;
    std::string_view response_in;
    std::string_view response_out;
};
```

The registry validates adjacent declared transforms for the same method/path.
If the previous module's output state does not match the next module's input
state, registration fails at server startup. Use `frontend_contracts::any` only
for modules that deliberately accept or preserve any state.

Current shared states are declared by the main audio.cpp repo in
`app/server/frontend.h`.

Listener modules register a factory with:

```cpp
registry.add_listener("websocket", make_websocket_listener);
```

The listener receives the host, port, core handler, shutdown callback, request
body limit, and string options map from the main server.
