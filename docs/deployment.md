# Deploy Fieldwise

Use the Linux Docker image for production. It runs Gunicorn with two workers as
UID 10001. The Flask and BrowserSync servers are for local development; see
[development.md](development.md). The bundled dataset and model are exploratory;
review their [data](data.md) and [evaluation](model.md) limitations before release.

## Build and start

Run commands from the repository root with Docker Engine running in Linux
container mode. Use a current Docker Compose that supports optional `env_file`
entries and `up --wait`. The image build needs network access for the pinned
Python base image and locked wheel dependencies.

Create a local configuration once:

```console
cp .env.example .env
```

In PowerShell use `Copy-Item .env.example .env`. Keep populated `.env` files out
of Git. Compose loads this file at runtime; the image build excludes it.

```console
docker compose config --quiet
docker compose build --pull
docker compose up -d --wait --wait-timeout 120
python -m scripts.smoke http://127.0.0.1:10000
```

The smoke command uses only Python's standard library and checks real pages,
assets, historical selection, inference, error responses and model metadata.
Compose validation checks configuration syntax; it does not prove that the
Docker daemon, build, container or application works.

The default address is `http://127.0.0.1:10000`. Compose binds to the host loopback
interface. A same-host reverse proxy can connect there. For a proxy on another
machine or container network, configure an appropriate reachable binding and
firewall so only that proxy can reach Gunicorn.

## Configuration and HTTPS

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIELDWISE_BIND_ADDRESS` | `127.0.0.1` | Compose host interface; changing it changes exposure. |
| `FIELDWISE_HOST_PORT` | `10000` | Compose host port; the container remains on port 10000. |
| `GUNICORN_CMD_ARGS` | See `.env.example` | Gunicorn workers, logs and timeouts. Keep the container bind on port 10000 for the image health check. |
| `FIELDWISE_LOG_LEVEL` | `INFO` | Application logging level; invalid levels prevent startup. |
| `FIELDWISE_HTTPS` | `false` | Adds the HSTS response header when enabled; does not terminate TLS or redirect HTTP. |
| `FIELDWISE_TRUST_PROXY` | `false` | Trusts one proxy hop for forwarded client address and protocol. |
| `FIELDWISE_TRUSTED_HOSTS` | Unset | Comma-separated allowed request hosts; omit the variable to leave host validation unrestricted. |
| `PORT` | `10000` | Direct Python/hosted process port; does not override the Docker Gunicorn bind. |

Terminate HTTPS at your hosting platform or reverse proxy and confirm that the
public HTTPS URL works before enabling `FIELDWISE_HTTPS=true`. Configure HTTP
redirects at that edge. Enable `FIELDWISE_TRUST_PROXY=true` only behind one
controlled proxy that overwrites forwarded headers and cannot be bypassed.
Only forwarded client address and protocol are trusted; forwarded host headers
are not used. Do not enable this option for direct client connections or assume
it supports an arbitrary chain of proxies.

Set `FIELDWISE_TRUSTED_HOSTS` to the public hostname and any intentional local
access hosts. Include `127.0.0.1` so the image's internal readiness check works;
include `localhost` if operators access that name. For example:

```dotenv
FIELDWISE_TRUSTED_HOSTS=fieldwise.example.org,127.0.0.1,localhost
```

Apply environment edits with `docker compose up -d --wait --wait-timeout 120`.
A container restart alone does not reload `.env`. Retest using the actual public
URL after proxy, host or TLS changes. Requests are limited to 16 KiB; an edge
proxy may enforce the same or a smaller bound.

## Readiness, logs and shutdown

```console
docker compose ps
docker compose logs --tail 100 web
docker compose logs -f web
docker compose down
```

`GET /health/live` returns `{"status":"alive"}` after application startup.
`GET /health/ready` returns `{"status":"ready"}` with HTTP 200 when the dataset
and predictor are loaded; unavailable resources produce HTTP 503. Readiness
does not run a prediction per request, so retain the full smoke check for
release verification. These endpoints cannot report startup failures before
Gunicorn has loaded the application; inspect container logs for those failures.

The image probes readiness on internal port 10000 every 30 seconds, allowing
30 seconds for initial startup. Compose uses a read-only root filesystem,
writable temporary `/tmp`, no Linux capabilities, no privilege escalation and a
PID limit. Gunicorn's temporary files and control socket use `/tmp`. The
35-second Compose stop grace period allows its 30-second graceful timeout.
Logs go to stdout/stderr; configure retention in the host logging system.

## Verify an image independently

```console
python -m scripts.container_smoke --image fieldwise:local
```

This creates a uniquely named disposable container, checks the API and image
health under production filesystem restrictions, verifies its nonroot user and
excluded files, then removes that container. Docker must be available. Failure
diagnostics retain the original error and report any log or cleanup failure;
if removal fails, remove only the named `fieldwise-smoke-*` container reported
in the exception after resolving the daemon problem.

The container CI workflow builds and executes this check on Linux. Backend CI
also runs deployment-tool regression tests on Windows and Linux. Passing those
mocked tests does not establish that a particular local image has been built
or started successfully.

## Artifacts, upgrades and rollback

The image contains the CSV, pickle, evaluation report and integrity manifest.
Startup verifies their matching hashes, feature schema and runtime compatibility
before deserializing the model. A bad or mixed release fails startup. Load only
trusted, reviewed files: hashes do not authenticate a malicious replacement of
both pickle and manifest.

For external artifacts, set `FIELDWISE_DATA_PATH`, `FIELDWISE_MODEL_PATH` and
`FIELDWISE_METRICS_PATH` to their container paths and mount files read-only.
`model_manifest.json` must be alongside the configured model. Publish the CSV
and all three model release files together; do not replace live files piecemeal.
Recreate the application to load a new release. See [model.md](model.md) for
training and integrity verification.

Before replacing an existing image, retain its image ID or a release tag and a
reviewed copy of the corresponding runtime configuration. For example, preserve
an existing `fieldwise:local` as `fieldwise:previous` before rebuilding it:

```console
docker image tag fieldwise:local fieldwise:previous
```

After a failed rollout, restore that release's runtime configuration, retag the
saved image and recreate without building:

```console
docker image tag fieldwise:previous fieldwise:local
docker compose up -d --force-recreate --wait --wait-timeout 120
python -m scripts.smoke http://127.0.0.1:10000
```

Use the configured host port in smoke URLs. Keep the prior image until the new
release passes readiness and smoke checks. No persistent user database or
application uploads need migration; externally mounted artifacts still require
the matching release to be restored.
