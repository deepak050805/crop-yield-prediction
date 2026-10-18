const { spawn, spawnSync } = require('node:child_process');
const { existsSync } = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const browserSync = require('browser-sync').create();

const root = path.resolve(__dirname, '..', '..');
const venvPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : 'python');
const backendPort = 5001;
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
    } else backend.kill();
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

async function start() {
  const deadline = Date.now() + 60000;
  while (!stopping) {
    const ready = await new Promise(resolve => {
      const request = http.get(`http://127.0.0.1:${backendPort}/`, response => {
        response.resume();
        resolve(response.statusCode === 200);
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
    listen: '127.0.0.1', port: 5173, open: false, ui: false, online: false,
    notify: false,
    files: ['frontend/templates/**/*.html', 'frontend/static/**/*'],
    reloadDebounce: 200,
  }, error => { if (error) { console.error(error.message); stop(1); } });
}

start().catch(error => { console.error(error.message); stop(1); });
