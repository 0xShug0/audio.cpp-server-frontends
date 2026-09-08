# WebSocket

`websocket` is a frontend listener capability, not a pre/post-processing module.
It serves normal HTTP routes and adds a WebSocket bridge to the same in-process
server handler.

## Build

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  -DAUDIOCPP_SERVER_FRONTENDS_DIR=/path/to/audio.cpp-server-frontends \
  -DAUDIOCPP_SERVER_FRONTEND_MODULES=websocket
```

The WebSocket capability uses the existing vendored `cpp-httplib` target. The
default server build does not include this dependency.

## Run

```bash
build/debug/bin/audiocpp_server \
  --ui \
  --ui-management \
  --frontend-listener websocket
```

The same fields can be set in `server.json`:

```json
{
  "frontend_listener": "websocket"
}
```

## Request Frames

Connect to the same endpoint path that would be used for HTTP. For example,
`ws://127.0.0.1:8080/health` calls `/health`, and
`ws://127.0.0.1:8080/v1/tasks/run` calls `/v1/tasks/run`.

Each text WebSocket message becomes one request to that endpoint. A raw JSON
message is forwarded as the endpoint's JSON body with method `POST`:

```json
{
  "model": "pocket-tts",
  "input": "Hello from audio.cpp!"
}
```

For GET routes or custom headers, use an envelope. The endpoint path still comes
from the WebSocket URL; frames cannot override it:

```json
{
  "id": "health-1",
  "method": "GET"
}
```

`body_json` and `body` are also accepted in the envelope:

```json
{
  "id": "run-1",
  "body_json": {
    "model": "pocket-tts",
    "input": "Hello from audio.cpp!"
  }
}
```

The response for a non-streaming route is:

```json
{
  "type": "response",
  "id": "health-1",
  "status": 200,
  "content_type": "application/json",
  "headers": {},
  "body": "{\"status\":\"ok\"}"
}
```

For a streaming route, the bridge sends `response.start`, one or more
`response.body` messages, and then `response.done`.
