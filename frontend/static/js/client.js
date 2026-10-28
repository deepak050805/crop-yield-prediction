(function (root) {
  'use strict';

  function csvCell(value) {
    let text = value == null ? '' : String(value);
    // Quoting alone does not stop spreadsheets from evaluating formulas.
    if (typeof value !== 'number' && /^[\s\u0000-\u001f]*[=+\-@]/u.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  }

  function historicalCsv(records) {
    const headers = ['District', 'Crop', 'Year', 'Temperature', 'Rainfall', 'Humidity', 'Yield'];
    return [headers, ...records.map(row => headers.map(key => row[key]))]
      .map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
  }

  const api = { csvCell, historicalCsv };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FieldwiseClient = api;
})(typeof window === 'object' ? window : globalThis);
