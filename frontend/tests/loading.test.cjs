const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { JSDOM } = require('jsdom');

const root = path.resolve(__dirname, '../..');
const venv = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.PYTHON || (fs.existsSync(venv) ? venv : 'python');
const rendered = spawnSync(python, ['-c', [
  'import json',
  'from backend.app import app',
  'client = app.test_client()',
  "pages = {p: client.get(p).get_data(as_text=True) for p in ['/', '/history', '/predictor']}",
  "records = client.post('/get_data', json={'crop':'Wheat','district':'Amritsar'}).get_json()",
  "image = str(app.jinja_env.get_template('skeletons.html').module.image('/test-image.svg', 37, 37))",
  "print(json.dumps({'pages':pages,'records':records,'metrics':client.get('/model_info').get_json(),'image':image}))",
].join('\n')], { cwd: root, encoding: 'utf8' });
assert.equal(rendered.status, 0, `Install backend/requirements.txt before running UI tests.\n${rendered.stderr}`);
const fixtures = JSON.parse(rendered.stdout);
const yieldText = (value, digits = 2) => Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
const expectedYield = yieldText(fixtures.records.reduce((sum, row) => sum + row.Yield, 0) / fixtures.records.length);
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const styles = ['dashboard', 'theme', 'skeletons'].map(name => read(`frontend/static/css/${name}.css`)).join('\n');
const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };

function setup(t, page = '/', { chartAvailable = true, chartThrows = false, testImage = false, cachedImage = false, search = '' } = {}) {
  const dom = new JSDOM(fixtures.pages[page], { url: `http://localhost${page}${search}`, runScripts: 'outside-only' });
  t.after(() => dom.window.close());
  const window = dom.window, document = window.document, requests = [], instances = new Map();
  const style = document.createElement('style'); style.textContent = styles; document.head.append(style);
  window.matchMedia = () => ({ matches: false });
  if (chartAvailable) window.Chart = class {
    static defaults = { font: {} };
    constructor(element, config) {
      if (chartThrows) throw new Error('Canvas unavailable');
      this.data = config.data; this.options = config.options; this.updates = 0;
      instances.set(element.id, [...(instances.get(element.id) || []), this]);
    }
    update() { this.updates++; }
    resize() {}
  };
  window.fetch = (url, options = {}) => new Promise((resolve, reject) => {
    // Keep aborted requests controllable to verify that stale responses are ignored.
    requests.push({ url, options, fail: message => reject(new Error(message)),
      respond: (data, status = 200) => resolve({ ok: status < 400, json: async () => data }) });
  });
  if (testImage) document.body.insertAdjacentHTML('beforeend', fixtures.image);
  const image = document.querySelector('[data-image] img');
  if (cachedImage) {
    Object.defineProperty(image, 'complete', { value: true });
    Object.defineProperty(image, 'naturalWidth', { value: 37 });
  }
  window.eval(read('frontend/static/js/loading.js'));
  window.eval(read('frontend/static/js/client.js'));
  window.eval(read('frontend/static/js/dashboard.js'));
  return {
    window, document, requests, instances,
    get: id => document.getElementById(id),
    take: url => requests.splice(requests.findIndex(request => request.url === url), 1)[0],
    change(id, value) {
      document.getElementById(id).value = value;
      document.getElementById(id).dispatchEvent(new window.Event('change', { bubbles: true }));
    },
    submit() { document.getElementById('predictionForm').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })); },
  };
}

test('fast responses replace skeletons immediately, and charts are reused on reload', async t => {
  const c = setup(t);
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'loading');
  assert.equal(c.instances.size, 0, 'Do not render empty charts during loading');
  c.take('/get_data').respond(fixtures.records);
  await flush();
  assert.ok([...c.document.querySelectorAll('[data-records]')].every(region => region.getAttribute('aria-busy') === 'false'));
  assert.equal(c.get('avgYield').textContent, expectedYield);
  const chart = c.instances.get('yieldChart')[0];
  c.change('crop', 'Rice');
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'loading');
  c.take('/get_data').respond(fixtures.records);
  await flush();
  assert.equal(c.instances.get('yieldChart').length, 1);
  assert.equal(chart.updates, 1);
});

test('empty and failed loads do not masquerade as skeletons or successful charts', async t => {
  const c = setup(t);
  c.take('/get_data').respond([]); await flush();
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'empty');
  assert.equal(c.instances.size, 0);
  assert.match(c.get('dataStatus').textContent, /No primary district records/);
  c.change('district', 'Ludhiana');
  c.take('/get_data').fail('Offline'); await flush();
  assert.ok([...c.document.querySelectorAll('[data-records]')].every(region => region.dataset.state === 'error' && region.getAttribute('aria-busy') === 'false'));
  assert.match(c.get('dataStatus').textContent, /Offline/);
});

test('outdated record requests are cancelled and cannot overwrite newer content', async t => {
  const c = setup(t), first = c.take('/get_data');
  c.change('district', 'Ludhiana');
  const second = c.take('/get_data');
  assert.equal(first.options.signal.aborted, true);
  first.respond(fixtures.records); await flush();
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'loading');
  second.respond(fixtures.records.map(row => ({ ...row, Yield: 8 }))); await flush();
  assert.equal(c.get('avgYield').textContent, yieldText(8));
});

test('a slow comparison does not block primary records, and table geometry stays reserved', async t => {
  const c = setup(t, '/history');
  c.take('/get_data').respond(fixtures.records); await flush();
  const tableSize = c.window.getComputedStyle(c.get('historyTableView')).blockSize;
  assert.ok(tableSize && tableSize !== 'auto', 'The table viewport must reserve its height');
  c.change('compareDistrict', 'Ludhiana');
  const comparison = c.take('/get_data'), primary = c.take('/get_data');
  assert.equal(JSON.parse(comparison.options.body).district, 'Ludhiana');
  primary.respond(fixtures.records); await flush();
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'success');
  assert.equal(c.get('historyRows').children.length, 10);
  assert.match(c.get('comparisonStatus').textContent, /Loading/);
  comparison.respond(fixtures.records.map(row => ({ ...row, District: 'Ludhiana' }))); await flush();
  assert.equal(c.get('historyRows').children.length, 20);
  assert.equal(c.window.getComputedStyle(c.get('historyTableView')).blockSize, tableSize);
  assert.equal(c.window.getComputedStyle(c.get('historyTableView').querySelector('table')).tableLayout, 'fixed');
  const tableRows = c.get('historyRows').firstChild;
  c.change('historyMeasure', 'Rainfall');
  assert.equal(c.get('historyRows').firstChild, tableRows, 'Changing the chart measure must not rebuild table rows');
});

test('comparison errors preserve successfully loaded primary records', async t => {
  const c = setup(t, '/history');
  c.take('/get_data').respond(fixtures.records); await flush();
  c.change('compareDistrict', 'Ludhiana');
  const comparison = c.take('/get_data'), primary = c.take('/get_data');
  primary.respond(fixtures.records); await flush();
  comparison.fail('Comparison offline'); await flush();
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'success');
  assert.equal(c.get('historyRows').children.length, 10);
  assert.match(c.get('comparisonStatus').textContent, /unavailable/);
});

test('model details load independently and provide a retry for failed requests', async t => {
  const c = setup(t, '/predictor');
  const primary = c.take('/get_data');
  const report = c.take('/model_info');
  const size = c.window.getComputedStyle(c.get('modelMetrics')).blockSize;
  assert.ok(size && size !== 'auto', 'The expanded report must reserve its height');
  report.fail('Report offline'); await flush();
  assert.equal(c.get('modelMetrics').dataset.state, 'error');
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'loading');
  c.get('modelMetrics').querySelector('button').click();
  assert.equal(c.get('modelMetrics').dataset.state, 'loading');
  c.take('/model_info').respond(fixtures.metrics); await flush();
  assert.equal(c.get('modelMetrics').dataset.state, 'success');
  assert.equal(c.window.getComputedStyle(c.get('modelMetrics')).blockSize, size);
  primary.respond(fixtures.records); await flush();
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'success');
});

test('prediction skeletons handle success, errors, cancellation, and late responses', async t => {
  const c = setup(t, '/predictor');
  c.take('/get_data').respond(fixtures.records); c.take('/model_info').respond(fixtures.metrics); await flush();
  c.submit();
  assert.equal(c.get('resultPanel').dataset.state, 'loading');
  assert.equal(c.get('comparisonPanel').hidden, false);
  const request = c.take('/predict');
  c.get('rainfallNumber').value = '750';
  c.get('rainfallNumber').dispatchEvent(new c.window.Event('input', { bubbles: true }));
  assert.equal(request.options.signal.aborted, true);
  assert.equal(c.get('resultPanel').dataset.state, 'idle');
  assert.equal(c.get('predictBtn').disabled, false);
  c.submit();
  const current = c.take('/predict');
  request.respond({ predicted: 99 }); await flush();
  assert.equal(c.get('resultPanel').dataset.state, 'loading');
  current.respond({ predicted: 3.4, optimal: 3.5, gap: .1, best_rain: 750, best_temp: 27, best_hum: 65, crop: 'Wheat', district: 'Amritsar', warnings: [] });
  await flush();
  assert.equal(c.get('resultPanel').dataset.state, 'success');
  assert.equal(c.get('predictedYield').textContent, yieldText(3.4, 3));
  c.submit(); c.take('/predict').fail('Prediction offline'); await flush();
  assert.equal(c.get('resultPanel').dataset.state, 'error');
  assert.equal(c.get('comparisonPanel').getAttribute('aria-busy'), 'false');
  assert.equal(c.get('predictionChart').parentElement.dataset.state, 'error');
  assert.equal(c.get('predictBtn').disabled, false);
});

test('cached images skip placeholders; image errors preserve dimensions and expose fallback', t => {
  const cached = setup(t, '/', { testImage: true, cachedImage: true });
  const cachedFrame = cached.document.querySelector('[data-image]');
  assert.equal(cachedFrame.dataset.state, 'success');
  cached.document.querySelector('.stat-card').append(cachedFrame);
  cached.change('crop', 'Rice');
  assert.equal(cachedFrame.querySelector('.loading-skeleton').hidden, true, 'A loading parent must not restart a cached image placeholder');
  const c = setup(t, '/', { testImage: true }), frame = c.document.querySelector('[data-image]'), image = frame.querySelector('img');
  const before = [c.window.getComputedStyle(frame).width, c.window.getComputedStyle(frame).height];
  assert.equal(image.getAttribute('width'), '37');
  assert.equal(image.getAttribute('height'), '37');
  image.dispatchEvent(new c.window.Event('error'));
  assert.equal(frame.dataset.state, 'error');
  assert.equal(frame.getAttribute('aria-busy'), 'false');
  assert.deepEqual([c.window.getComputedStyle(frame).width, c.window.getComputedStyle(frame).height], before);
  assert.ok(frame.querySelector('.image-fallback'));
});

test('themes, reduced motion, placeholder accessibility, and missing charts are handled', async t => {
  const c = setup(t, '/', { chartAvailable: false });
  const light = c.window.getComputedStyle(c.document.documentElement).getPropertyValue('--skeleton-base');
  c.document.documentElement.dataset.theme = 'dark';
  const dark = c.window.getComputedStyle(c.document.documentElement).getPropertyValue('--skeleton-base');
  assert.notEqual(light, dark);
  assert.ok([...c.document.querySelectorAll('.loading-skeleton')].every(element => element.getAttribute('aria-hidden') === 'true'));
  const rules = [...c.document.styleSheets[0].cssRules];
  assert.ok(rules.some(rule => rule.conditionText?.replaceAll(' ', '') === '(prefers-reduced-motion:reduce)' && rule.cssText.includes('skeleton-surface::after') && rule.cssText.includes('animation: none')));
  c.take('/get_data').respond(fixtures.records); await flush();
  assert.equal(c.get('yieldChart').parentElement.dataset.state, 'error');
  assert.equal(c.get('avgYield').textContent, expectedYield);
});

test('historical charts preserve uneven year spacing and display measurement units', async t => {
  const c = setup(t, '/history');
  const records = [fixtures.records[0], fixtures.records.at(-1)];
  c.take('/get_data').respond(records); await flush();
  const chart = c.instances.get('historyChart')[0];
  assert.equal(chart.options.scales.x.type, 'linear');
  assert.equal(chart.options.scales.x.title.text, 'Year');
  assert.deepEqual(Array.from(chart.data.datasets[0].data, point => point.x), records.map(row => row.Year));
  c.change('historyMeasure', 'Rainfall');
  const label = chart.options.plugins.tooltip.callbacks.label({ dataset: { label: 'Amritsar' }, parsed: { y: 750 } });
  assert.match(label, /750.*mm/);
  assert.equal(chart.options.scales.y.title.text, 'Rainfall (mm)');
});

test('malformed API records fail safely without partial tables or NaN charts', async t => {
  const c = setup(t, '/history');
  c.take('/get_data').respond([{ ...fixtures.records[0], Humidity: null }]); await flush();
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'error');
  assert.equal(c.get('historyRows').children.length, 0);
  assert.equal(c.get('downloadCsv').disabled, true);
  assert.match(c.get('dataStatus').textContent, /unexpected format/);
});

test('unavailable R squared is not presented as a measured zero', async t => {
  const c = setup(t, '/predictor');
  const metrics = JSON.parse(JSON.stringify(fixtures.metrics));
  metrics.holdout.r2 = null;
  metrics.per_crop.Wheat.r2 = null;
  c.take('/model_info').respond(metrics); await flush();
  assert.equal(c.get('modelMetrics').dataset.state, 'success');
  assert.match(c.get('modelMetrics').textContent, /Overall R²: unavailable; Wheat R²: unavailable/);
});

test('historical shared links restore comparison and measure without extra measure requests', async t => {
  const c = setup(t, '/history', { search: '?compare=Ludhiana&measure=Rainfall' });
  assert.equal(c.get('compareDistrict').value, 'Ludhiana');
  assert.equal(c.get('historyMeasure').value, 'Rainfall');
  assert.equal(JSON.parse(c.take('/get_data').options.body).district, 'Ludhiana');
  c.take('/get_data').respond(fixtures.records); await flush();
  assert.equal(c.instances.get('historyChart')[0].options.scales.y.title.text, 'Rainfall (mm)');
  c.change('historyMeasure', 'Humidity');
  assert.equal(c.requests.length, 0);
  assert.equal(new URL(c.window.location.href).searchParams.get('measure'), 'Humidity');
  assert.ok([...c.document.querySelectorAll('[data-nav]')].every(link =>
    new URL(link.href).searchParams.get('compare') === 'Ludhiana'));
  c.change('compareDistrict', '');
  assert.equal(new URL(c.window.location.href).searchParams.has('compare'), false);
  const invalid = setup(t, '/history', { search: '?compare=unknown&measure=unknown' });
  assert.equal(invalid.get('compareDistrict').value, '');
  assert.equal(invalid.get('historyMeasure').value, 'Yield');
});

test('broken chart construction preserves exportable tables and numerical estimates', async t => {
  const c = setup(t, '/history', { chartThrows: true });
  c.take('/get_data').respond(fixtures.records); await flush();
  assert.equal(c.get('historyChartView').dataset.state, 'error');
  assert.equal(c.document.querySelector('[data-records]').dataset.state, 'success');
  assert.equal(c.get('historyRows').children.length, fixtures.records.length);
  assert.equal(c.get('downloadCsv').disabled, false);
  const prediction = setup(t, '/predictor', { chartThrows: true });
  prediction.take('/get_data').respond(fixtures.records);
  prediction.take('/model_info').respond(fixtures.metrics); await flush();
  prediction.submit();
  prediction.take('/predict').respond({ predicted: 3.4, optimal: 3.5, gap: .1, best_rain: 750,
    best_temp: 27, best_hum: 65, crop: 'Wheat', district: 'Amritsar', warnings: [] });
  await flush();
  assert.equal(prediction.get('resultPanel').dataset.state, 'success');
  assert.equal(prediction.get('predictedYield').textContent, yieldText(3.4, 3));
  assert.equal(prediction.get('predictionChart').parentElement.dataset.state, 'error');
});

test('a failed chart update can recover without refetching or losing records', async t => {
  const c = setup(t, '/history');
  c.take('/get_data').respond(fixtures.records); await flush();
  const chart = c.instances.get('historyChart')[0];
  let destroyed = false;
  chart.update = () => { throw new Error('Rendering failed'); };
  chart.destroy = () => { destroyed = true; };
  c.change('historyMeasure', 'Rainfall');
  assert.equal(destroyed, true);
  assert.equal(c.get('historyChartView').dataset.state, 'error');
  assert.equal(c.get('historyRows').children.length, fixtures.records.length);
  c.change('historyMeasure', 'Humidity');
  assert.equal(c.instances.get('historyChart').length, 2);
  assert.equal(c.get('historyChartView').dataset.state, 'success');
  assert.equal(c.requests.length, 0);
});
