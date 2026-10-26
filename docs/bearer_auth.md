# bearer_auth

`bearer_auth` is an access-control pre-processing module. It is the
authenticated gateway for deployments that expose `audiocpp_server` beyond
localhost, without adding authentication to the core server:

```text
pre:
* *
any -> any
```

For every request it either passes the request through unchanged or stops it
before the core handler:

1. A request matching a public route passes without a token. By default the
   only public route is `GET /health`.
2. Any other request needs `Authorization: Bearer <token>`. A missing or
   malformed header, or a wrong token, gets `401` with a `WWW-Authenticate:
   Bearer` challenge.
3. With a valid token, a request matching the deny list, or missing from the
   allow list when one is configured, gets `403`.

The token is compared in constant time with respect to its content and the
presented token's length. It is never logged or echoed in responses.

CORS preflight (`OPTIONS`) is answered by the core before frontend modules
run, so browsers can still send the authenticated request that follows.

## Build

List `bearer_auth` first so that later modules never process an
unauthenticated request:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  -DAUDIOCPP_SERVER_FRONTENDS_DIR=/path/to/audio.cpp-server-frontends \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=bearer_auth;audio_decode;mp3_encode'
```

## Configuration

Pre-processing modules do not receive `--frontend-option` values (those go
to the selected listener only), so the module reads environment variables
once at server startup:

| Variable | Default | Meaning |
|---|---|---|
| `AUDIOCPP_BEARER_AUTH_TOKEN` | required | Shared bearer token. The server refuses to start if it is unset or empty. |
| `AUDIOCPP_BEARER_AUTH_PUBLIC_ROUTES` | `GET /health` | Routes reachable without a token. Set it to an empty value to require the token everywhere. |
| `AUDIOCPP_BEARER_AUTH_ALLOW_ROUTES` | unset | If set, only these routes are permitted with a valid token. |
| `AUDIOCPP_BEARER_AUTH_DENY_ROUTES` | unset | Routes refused even with a valid token. Checked before the allow list. |

Route lists are comma-separated `METHOD /path` entries. `*` as the method
matches any method, and a path ending in `*` matches by prefix. Matching is
otherwise exact and case-sensitive, the same comparison the core dispatcher
uses to select a handler, so a request that does not match an entry exactly
cannot reach that entry's handler either. The query string is not part of
the path.

## Remote deployment

Allow the inference API and model load/unload/upload, while denying the
server-side directory browser, models-root changes and the WebUI:

```bash
export AUDIOCPP_BEARER_AUTH_TOKEN="$(cat /run/secrets/audiocpp_token)"
export AUDIOCPP_BEARER_AUTH_DENY_ROUTES='POST /v1/ui/browse-directories,POST /v1/ui/models-root,GET /,GET /index.html'
build/debug/bin/audiocpp_server --config server.json --host 0.0.0.0
```

Or permit only an explicit set of routes:

```bash
export AUDIOCPP_BEARER_AUTH_ALLOW_ROUTES='GET /v1/models,POST /v1/models/load,POST /v1/models/unload,POST /v1/ui/upload,POST /v1/audio/*,POST /v1/tasks/*'
```

A bearer token is only as private as the transport. Combine this module with
the [`https`](https.md) listener, or terminate TLS in front of the server,
when the network between client and server is not trusted.

## Request

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $AUDIOCPP_BEARER_AUTH_TOKEN"
```
