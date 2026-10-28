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
