const { spawn, spawnSync } = require('node:child_process');
const { existsSync } = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const root = path.resolve(__dirname, '..', '..');
const venvPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : 'python');
function readPort(name, fallback) {
  const value = process.env[name] ?? String(fallback);
  const parsed = /^\d+$/.test(value) ? Number(value) : NaN;
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) {
    throw new Error(`${name} must be an integer from 1 to 65535.`);
  }
  return parsed;
}

const backendPort = readPort('FIELDWISE_BACKEND_PORT', 5001);
const frontendPort = readPort('FIELDWISE_DEV_PORT', 5173);
if (backendPort === frontendPort) {
  throw new Error('FIELDWISE_BACKEND_PORT and FIELDWISE_DEV_PORT must be different.');
}
const browserSync = require('browser-sync').create();
let stopping = false;

const backend = spawn(python, [
  '-m', 'flask', '--app', 'backend.app', 'run', '--debug', '--no-debugger',
  '--no-reload', '--host', '127.0.0.1', '--port', String(backendPort),
], {
  cwd: root, stdio: 'inherit', windowsHide: true,
  env: { ...process.env, FLASK_SKIP_DOTENV: '1', PYTHONUNBUFFERED: '1' },
});

function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  browserSync.exit();
  if (backend.exitCode === null && backend.pid) {
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/pid', String(backend.pid), '/t', '/f'], { stdio: 'ignore', windowsHide: true });
    } else {
      backend.kill('SIGTERM');
      const forceStop = setTimeout(() => backend.kill('SIGKILL'), 5000);
      forceStop.unref();
      backend.once('exit', () => clearTimeout(forceStop));
    }
  }
  process.exitCode = code;
}

backend.on('error', error => {
  console.error(`Could not start Python: ${error.message}. Create .venv and install backend/requirements.txt first.`);
  stop(1);
});
backend.on('exit', code => { if (!stopping) stop(code || 1); });
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
process.once('uncaughtException', error => { console.error(error); stop(1); });
process.once('unhandledRejection', error => { console.error(error); stop(1); });

async function start() {
  const deadline = Date.now() + 60000;
  while (!stopping) {
    const ready = await new Promise(resolve => {
      const request = http.get(`http://127.0.0.1:${backendPort}/health/ready`, response => {
        let body = '';
        response.setEncoding('utf8');
        response.on('data', chunk => {
          body += chunk;
          if (body.length > 1024) response.destroy(new Error('Unexpected readiness response.'));
        });
        response.on('error', () => resolve(false));
        response.on('end', () => {
          try { resolve(response.statusCode === 200 && JSON.parse(body).status === 'ready'); }
          catch { resolve(false); }
        });
      });
      request.setTimeout(1000, () => request.destroy());
      request.on('error', () => resolve(false));
    });
    if (stopping) return;
    if (ready) break;
    if (Date.now() >= deadline) {
      console.error('Flask did not become ready. Check the Python output above.');
      stop(1);
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  if (stopping) return;
  browserSync.init({
    proxy: `http://127.0.0.1:${backendPort}`,
    listen: '127.0.0.1', port: frontendPort, open: false, ui: false, online: false,
    notify: false,
    files: ['frontend/templates/**/*.html', 'frontend/static/**/*'],
    reloadDebounce: 200,
  }, error => { if (error) { console.error(error.message); stop(1); } });
}

start().catch(error => { console.error(error.message); stop(1); });
