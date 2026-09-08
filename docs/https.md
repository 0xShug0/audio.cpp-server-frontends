# HTTPS

`https` is a frontend listener capability, not a pre/post-processing module. It
serves the same in-process server over HTTPS and forwards requests to the same
core handler used by the plain HTTP listener.

## Build

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  -DAUDIOCPP_SERVER_FRONTENDS_DIR=/path/to/audio.cpp-server-frontends \
  -DAUDIOCPP_SERVER_FRONTEND_MODULES=https
```

The HTTPS capability uses the existing vendored `cpp-httplib` target. That
target uses bundled BoringSSL by default, or system OpenSSL when configured with
`-DAUDIOCPP_USE_SYSTEM_OPENSSL=ON`.

The default server build does not include this TLS dependency.

## Run

```bash
build/debug/bin/audiocpp_server \
  --config server.json \
  --frontend-listener https \
  --frontend-option cert_file=cert.pem \
  --frontend-option key_file=key.pem
```

The same fields can be set in `server.json`:

```json
{
  "frontend_listener": "https",
  "frontend_options": {
    "cert_file": "cert.pem",
    "key_file": "key.pem"
  }
}
```

When HTTPS is enabled at runtime, that listener port is HTTPS-only. Plain HTTP
requests sent to the HTTPS port fail before reaching the server handler.

`cert_file` and `key_file` are listener options and are used as provided by the
HTTPS frontend. Relative paths are resolved from the server process working
directory.
