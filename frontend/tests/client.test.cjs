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
