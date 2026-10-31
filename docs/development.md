# Develop Fieldwise locally

Run commands from the repository root. Use Python 3.12 and Node.js 18 or newer;
frontend CI currently checks Node 22 and 24. Docker is optional for development.

```console
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then install the locked tools:

```console
python -m pip install -r backend/requirements-dev.txt
python -m pip check
npm ci --ignore-scripts
npm run dev
```

Open the local BrowserSync address printed in the terminal, normally
`http://localhost:5173`. Flask listens on loopback port 5001. The launcher waits
for `/health/ready` before starting the proxy; if Flask fails to start, read its
terminal output. BrowserSync reloads template/JavaScript edits and injects CSS
updates. Restart the command after Python changes or model replacement because
the Flask code reloader is disabled. Press Ctrl+C to stop both processes.

The launcher uses `.venv` when present, otherwise `python` from PATH. Set `PYTHON`
to an executable path to override it. Its configurable ports must be distinct
integers from 1 to 65535:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIELDWISE_BACKEND_PORT` | `5001` | Flask port used by the live reload proxy. |
| `FIELDWISE_DEV_PORT` | `5173` | Requested BrowserSync port; inspect its printed URL if occupied. |

For example, in PowerShell:

```powershell
$env:FIELDWISE_BACKEND_PORT = '5002'
$env:FIELDWISE_DEV_PORT = '5174'
npm run dev
```

The launcher deliberately skips dotenv loading. Set application configuration in
the process environment; merely editing `.env` does not configure this command.
See [deployment.md](deployment.md) and [.env.example](../.env.example) for
application settings and trusted artifact paths. Leave proxy trust and HTTPS
flags at their defaults for direct local development.

To run Flask without live reload:

```console
python -m backend.app
```

This uses loopback port 10000 by default, configurable with `PORT`. Gunicorn is
installed only on Linux; use Flask for local Windows development and the Linux
image for production.

## Checks used by CI

```console
python -m pip check
python -m ruff check backend scripts
python -m mypy backend/config.py backend/validation.py
python -m unittest discover -s backend/tests -v
python -m unittest discover -s scripts/tests -v
npm run test:ui
```

UI tests render actual Flask templates and therefore require the Python runtime
dependencies as well as npm dependencies. Backend tests load the bundled model;
keep the locked scikit-learn version and matching model manifest. The deployment
tool tests use an ephemeral local HTTP server and mocked Docker interactions.
For a running app, exercise the actual API with:

```console
python -m scripts.smoke http://127.0.0.1:10000
```

Use port 5001 instead when verifying the Flask process started by `npm run dev`.
Full image verification requires a running Linux Docker daemon and is described
in [deployment.md](deployment.md). Train into a separate output directory when
experimenting, following [model.md](model.md), so a partial experiment does not
replace the bundled release used by application tests.
