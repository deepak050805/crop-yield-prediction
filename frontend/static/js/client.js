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

  async function requestJson(url, options = {}, { timeoutMs = 20000, fetch = root.fetch.bind(root) } = {}) {
    const controller = new AbortController();
    const cancel = () => controller.abort();
    let timedOut = false;
    if (options.signal?.aborted) controller.abort();
    options.signal?.addEventListener('abort', cancel, { once: true });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      let data;
      try { data = await response.json(); }
      catch { throw new Error('The server returned an unreadable response. Please try again.'); }
      if (!response.ok) {
        throw new Error(typeof data?.error === 'string' ? data.error : `The request failed (HTTP ${response.status}). Please try again.`);
      }
      return data;
    } catch (error) {
      if (timedOut) throw new Error('The request timed out. Please try again.');
      throw error;
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener('abort', cancel);
    }
  }

  const api = { csvCell, historicalCsv, requestJson };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FieldwiseClient = api;
})(typeof window === 'object' ? window : globalThis);
