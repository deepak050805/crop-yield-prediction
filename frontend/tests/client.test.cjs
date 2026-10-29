const { test } = require('node:test');
const assert = require('node:assert/strict');
const client = require('../static/js/client.js');

test('CSV text cannot execute spreadsheet formulas, including leading whitespace', () => {
  for (const value of ['=1+1', '+SUM(A1)', '-2+3', '@SUM(A1)', ' \t=1', '\r\n+2']) {
    assert.ok(client.csvCell(value).startsWith('"\''), value);
  }
  assert.equal(client.csvCell(-20), '"-20"', 'Negative weather measurements remain numeric');
  assert.equal(client.csvCell('Wheat, "winter"'), '"Wheat, ""winter"""');
  assert.equal(client.csvCell(null), '""');
});

test('historical CSV has stable columns, escaped text, and RFC 4180 line endings', () => {
  const result = client.historicalCsv([{ District: 'A, B', Crop: 'Rice', Year: 2024,
    Temperature: -2, Rainfall: 150, Humidity: 75, Yield: 3.2 }]);
  assert.equal(result, '"District","Crop","Year","Temperature","Rainfall","Humidity","Yield"\r\n' +
    '"A, B","Rice","2024","-2","150","75","3.2"\r\n');
});

test('requests surface structured errors and handle non-JSON reverse proxy responses', async () => {
  await assert.rejects(client.requestJson('/predict', {}, {
    fetch: async () => ({ ok: false, status: 400, json: async () => ({ error: 'Choose a crop.' }) }),
  }), /Choose a crop/);
  await assert.rejects(client.requestJson('/predict', {}, {
    fetch: async () => ({ ok: false, status: 502, json: async () => { throw new SyntaxError('HTML'); } }),
  }), /unreadable response/);
  await assert.rejects(client.requestJson('/predict', {}, {
    fetch: async () => ({ ok: false, status: 503, json: async () => ({ error: {} }) }),
  }), /HTTP 503/);
});

test('requests time out and preserve cancellation by callers', async () => {
  const fetch = (url, { signal }) => new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException('Aborted', 'AbortError'));
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  });
  await assert.rejects(client.requestJson('/slow', {}, { fetch, timeoutMs: 5 }), /timed out/);
  const controller = new AbortController();
  const request = client.requestJson('/cancel', { signal: controller.signal }, { fetch });
  controller.abort();
  await assert.rejects(request, error => error.name === 'AbortError');
  await assert.rejects(client.requestJson('/already-cancelled', { signal: controller.signal }, { fetch }), error => error.name === 'AbortError');
});

test('cancelling response body parsing preserves AbortError and releases the request timer', async () => {
  const controller = new AbortController();
  const aborted = new DOMException('Body read cancelled', 'AbortError');
  const request = client.requestJson('/stream', { signal: controller.signal }, {
    fetch: async (url, { signal }) => ({ ok: true, json: () => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(aborted), { once: true });
    }) }),
  });
  await Promise.resolve();
  controller.abort();
  await assert.rejects(request, error => error === aborted);
});

test('prediction responses validate the complete scenario before display', () => {
  const valid = { predicted: 3.4, optimal: 3.5, gap: 0.1, best_rain: 750,
    best_temp: 27, best_hum: 65, crop: 'Wheat', district: 'Amritsar', warnings: [] };
  assert.equal(client.validatePrediction(valid, 'Wheat', 'Amritsar'), valid);
  for (const override of [null, { best_rain: undefined }, { best_temp: NaN }, { best_hum: 101 },
    { predicted: '3.4' }, { warnings: 'warning' }, { warnings: [{}] }, { crop: 'Rice' },
    { district: 'Ludhiana' }, { gap: -1 }, { gap: 1 }, { optimal: 3 }]) {
    assert.throws(() => client.validatePrediction(override === null ? null : { ...valid, ...override }, 'Wheat', 'Amritsar'), /invalid estimate/);
  }
});

test('historical records reject malformed measurements and sort a copy chronologically', () => {
  const row = { Year: 2024, District: 'Amritsar', Crop: 'Wheat', Yield: 3.5,
    Temperature: 27, Rainfall: 750, Humidity: 65 };
  const rows = [row, { ...row, Year: 2023 }];
  assert.deepEqual(client.validateRecords(rows).map(value => value.Year), [2023, 2024]);
  assert.equal(rows[0].Year, 2024, 'The API response must not be mutated');
  assert.deepEqual(client.validateRecords([]), []);
  for (const data of [null, {}, [null], [{ ...row, Yield: NaN }], [{ ...row, Rainfall: '750' }],
    [{ ...row, Year: 2024.5 }], [{ ...row, District: '' }], [{ ...row, Crop: null }]]) {
    assert.throws(() => client.validateRecords(data), /unexpected format/);
  }
});

test('published model metrics reject incomplete scores and allow undefined R squared as null', () => {
  const score = { mae: .1, r2: null };
  const valid = { model: 'Ridge', holdout_years: [2023, 2024], holdout: score,
    baseline_holdout: score, per_crop: { Wheat: score } };
  assert.equal(client.validateMetrics(valid), valid);
  for (const override of [{ holdout_years: [] }, { model: null }, { baseline_holdout: null },
    { holdout: { mae: -1, r2: .5 } }, { per_crop: [] }, { per_crop: { Wheat: { mae: '0.1' } } }]) {
    assert.throws(() => client.validateMetrics({ ...valid, ...override }), /unexpected format/);
  }
});
