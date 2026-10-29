(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const page = document.body.dataset.page;
  const charts = new Map();
  const failedCharts = new Set();
  const { setState } = window.FieldwiseLoading;
  const { requestJson } = window.FieldwiseClient;
  const fields = ['rainfall', 'temperature', 'humidity'];
  const units = { Yield: 't/ha', Rainfall: 'mm', Temperature: '°C', Humidity: '%' };
  const columns = { rainfall: 'Rainfall', temperature: 'Temperature', humidity: 'Humidity' };
  let rows = [], comparison = [], allRows = [], metrics = null;
  let loadVersion = 0, predictionVersion = 0, hasPrediction = false, firstLoad = true;
  let recordsController = null, predictionController = null, modelController = null;
  let modelSkeleton = null;
  let metricsCrop = null;
  const crop = () => byId('crop').value;
  const district = () => byId('district').value;
  const format = (number, digits = 1) => Number(number).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
  const mean = (data, key) => data.reduce((sum, row) => sum + Number(row[key]), 0) / data.length;
  const scoreText = value => Number.isFinite(value) ? format(value, 3) : 'unavailable';

  if (window.Chart) {
    Chart.defaults.font.family = "'DM Sans', sans-serif";
    Chart.defaults.font.size = 10;
    Chart.defaults.color = chartPalette().text;
    Chart.defaults.animation = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : { duration: 450 };
  }

  function notice(id, message, error = false) {
    const element = byId(id);
    if (!element) return;
    element.textContent = message;
    element.hidden = !message;
    element.classList.toggle('error', error);
    if (id === 'dataStatus' && byId('retryData')) byId('retryData').hidden = !error;
  }

  function setRecordsState(state) {
    document.querySelectorAll('[data-records]').forEach(element => {
      setState(element, state);
    });
    if (page === 'history') setState(byId('historyChartView'), state);
  }

  function chartPalette() {
    const styles = getComputedStyle(document.documentElement);
    const color = (name, fallback) => styles.getPropertyValue(`--chart-${name}`).trim() || fallback;
    return {
      text: color('text', '#819078'), grid: color('grid', '#edf0e7'),
      primary: color('primary', '#537b4d'), secondary: color('secondary', '#ba9654'),
      fill: color('fill', '#dce8cf70'), point: color('point', '#fff'),
      scatter: color('scatter', '#759251'), bar: color('bar', '#386646'),
      barSoft: color('bar-soft', '#aebd83'), tooltip: color('tooltip', '#264c36'),
      tooltipText: color('tooltip-text', '#fff'),
    };
  }

  function themeDatasets(id, datasets, colors) {
    return datasets.map((dataset, index) => {
      if (id === 'predictionChart') return { ...dataset, backgroundColor: [colors.barSoft, colors.bar] };
      if (id === 'rainChart') return { ...dataset, backgroundColor: colors.scatter };
      const color = index ? colors.secondary : colors.primary;
      return { ...dataset, borderColor: color, backgroundColor: dataset.fill ? colors.fill : color,
        pointBackgroundColor: colors.point };
    });
  }

  window.addEventListener('themechange', () => {
    if (!window.Chart) return;
    const colors = chartPalette();
    Chart.defaults.color = colors.text;
    charts.forEach((chart, id) => {
      chart.data.datasets = themeDatasets(id, chart.data.datasets, colors);
      chart.options.color = colors.text;
      chart.options.plugins.legend.labels.color = colors.text;
      Object.assign(chart.options.plugins.tooltip, {
        backgroundColor: colors.tooltip, titleColor: colors.tooltipText, bodyColor: colors.tooltipText,
      });
      for (const axis of ['x', 'y']) {
        chart.options.scales[axis].ticks.color = colors.text;
        chart.options.scales[axis].title.color = colors.text;
        chart.options.scales[axis].grid.color = colors.grid;
      }
      chart.update('none');
    });
  });

  function draw(id, type, labels, datasets, axisTitles = {}) {
    if (!byId(id)) return;
    if (!window.Chart) { failedCharts.add(id); setState(byId(id).parentElement, 'error'); return; }
    const colors = chartPalette();
    const config = {
      type,
      data: { labels, datasets: themeDatasets(id, datasets, colors) },
      options: {
        responsive: true, maintainAspectRatio: false, color: colors.text,
        interaction: { mode: type === 'scatter' ? 'nearest' : 'index', intersect: false },
        plugins: {
          legend: { display: id === 'historyChart', position: 'bottom', labels: { color: colors.text, usePointStyle: true, boxWidth: 7, padding: 22 } },
          tooltip: { backgroundColor: colors.tooltip, titleColor: colors.tooltipText, bodyColor: colors.tooltipText, padding: 12, cornerRadius: 8,
            callbacks: {
              label: context => type === 'scatter'
                ? `Rainfall: ${format(context.parsed.x)} mm; Yield: ${format(context.parsed.y, 3)} t/ha`
                : `${context.dataset.label}: ${format(context.parsed.y, 3)} ${axisTitles.unit || 't/ha'}`,
              ...(type === 'scatter' ? { title: items => `Year: ${items[0]?.raw.year ?? 'unknown'}` } : {}),
            } },
        },
        scales: {
          x: { ...(id === 'historyChart' || id === 'yieldChart' ? { type: 'linear' } : {}),
            grid: { display: false, color: colors.grid }, border: { display: false }, ticks: { color: colors.text, maxTicksLimit: 8, precision: 0 },
            title: { display: Boolean(axisTitles.x), text: axisTitles.x, color: colors.text } },
          y: { grid: { color: colors.grid }, border: { display: false },
            beginAtZero: type === 'bar', ticks: { color: colors.text, maxTicksLimit: 5 },
            title: { display: Boolean(axisTitles.y), text: axisTitles.y, color: colors.text } },
        },
      },
    };
    try {
      if (charts.has(id)) {
        const chart = charts.get(id);
        chart.data = config.data;
        chart.options = config.options;
        chart.update();
      } else charts.set(id, new Chart(byId(id), config));
    } catch {
      const failed = charts.get(id) || Chart.getChart?.(byId(id));
      try { failed?.destroy?.(); } catch { /* Preserve the records if chart cleanup also fails. */ }
      charts.delete(id);
      failedCharts.add(id);
      setState(byId(id).parentElement, 'error');
      return;
    }
    failedCharts.delete(id);
    setState(byId(id).parentElement, 'success');
  }

  function line(label, values, color = '#537b4d', fill = false) {
    return { label, data: values, borderColor: color, backgroundColor: fill ? '#dce8cf70' : color,
      fill, borderWidth: 2, tension: 0.25, pointRadius: 3, pointHoverRadius: 6,
      pointBackgroundColor: '#fff', pointBorderWidth: 2 };
  }

  async function records(selectedCrop, selectedDistrict, signal) {
    const data = await requestJson('/get_data', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ crop: selectedCrop, district: selectedDistrict }),
      signal,
    });
    return window.FieldwiseClient.validateRecords(data);
  }

  function inPeriod(data) {
    if (!byId('fromYear')) return data;
    const from = Number(byId('fromYear').value), to = Number(byId('toYear').value);
    return data.filter(row => row.Year >= from && row.Year <= to);
  }

  function updateNavigation() {
    const query = new URLSearchParams(location.search);
    query.set('crop', crop()); query.set('district', district());
    if (byId('fromYear')) {
      query.set('from', byId('fromYear').value);
      query.set('to', byId('toYear').value);
    }
    if (byId('compareDistrict')) {
      const compare = byId('compareDistrict').value;
      if (compare) query.set('compare', compare);
      else query.delete('compare');
      query.set('measure', byId('historyMeasure').value);
    }
    history.replaceState(null, '', `${location.pathname}?${query}`);
    document.querySelectorAll('[data-nav]').forEach(link => {
      const url = new URL(link.href);
      url.search = query.toString();
      link.href = url.toString();
    });
  }

  function markStale() {
    predictionVersion += 1;
    const wasPredicting = Boolean(predictionController);
    predictionController?.abort();
    predictionController = null;
    if (wasPredicting) finishPrediction();
    if (!hasPrediction && !wasPredicting) return;
    byId('resultBadge').textContent = 'Inputs changed';
    notice('predictionStatus', 'Your inputs have changed. Select Predict yield to update this estimate.');
  }

  function renderOverview() {
    for (const [id, key, digits] of [['avgYield', 'Yield', 2], ['avgRain', 'Rainfall', 0],
      ['avgTemp', 'Temperature', 1], ['avgHum', 'Humidity', 0]]) {
      byId(id).textContent = rows.length ? format(mean(rows, key), digits) : '—';
    }
    byId('summaryPeriod').textContent = rows.length ? `${rows[0].Year}–${rows.at(-1).Year} · ${rows.length} records` : 'No matching records';
    if (rows.length) {
      draw('yieldChart', 'line', rows.map(row => row.Year), [line('Yield', rows.map(row => ({ x: row.Year, y: row.Yield })), '#537b4d', true)], { x: 'Year', y: 'Yield (t/ha)' });
      draw('rainChart', 'scatter', [], [{ label: 'Rainfall vs yield',
      data: rows.map(row => ({ x: row.Rainfall, y: row.Yield, year: row.Year })), backgroundColor: '#759251', pointRadius: 5, pointHoverRadius: 7 }],
      { x: 'Rainfall (mm)', y: 'Yield (t/ha)' });
    }
    byId('insightTitle').textContent = `${crop()} in ${district()}`;
    if (!rows.length) {
      byId('overviewInsight').textContent = 'Choose another year range to explore historical performance.';
      return;
    }
    const best = rows.reduce((previous, current) => current.Yield > previous.Yield ? current : previous);
    byId('overviewInsight').textContent = `Highest recorded yield in this selection: ${format(best.Yield, 2)} t/ha in ${best.Year}. Average across ${rows.length} seasons: ${format(mean(rows, 'Yield'), 2)} t/ha.`;
  }

  function renderHistory(updateTable = true) {
    const measure = byId('historyMeasure').value;
    const datasets = [line(district(), rows.map(row => ({ x: row.Year, y: row[measure] })))];
    if (comparison.length) datasets.push(line(byId('compareDistrict').value, comparison.map(row => ({ x: row.Year, y: row[measure] })), '#ba9654'));
    const years = [...new Set([...rows, ...comparison].map(row => row.Year))].sort((a, b) => a - b);
    if (years.length) draw('historyChart', 'line', years, datasets, { x: 'Year', y: `${measure} (${units[measure]})`, unit: units[measure] });
    byId('historyTitle').textContent = `${measure} through the years`;
    if (!updateTable) return;
    const displayRows = [...rows, ...comparison].sort((a, b) => a.Year - b.Year || a.District.localeCompare(b.District));
    byId('historyRows').replaceChildren(...displayRows.map(row => {
      const tr = document.createElement('tr');
      for (const key of ['Year', 'District', 'Crop', 'Yield', 'Rainfall', 'Temperature', 'Humidity']) {
        const td = document.createElement('td');
        td.textContent = row[key];
        tr.append(td);
      }
      return tr;
    }));
    byId('recordCount').textContent = `${displayRows.length} records · ${crop()}`;
    byId('downloadCsv').disabled = !displayRows.length;
  }

  function fillAverages() {
    if (!allRows.length) return;
    for (const field of fields) {
      const value = mean(allRows, columns[field]).toFixed(field === 'rainfall' ? 0 : 1);
      byId(`${field}Number`).value = value;
      byId(`${field}Range`).value = value;
    }
    markStale();
  }

  async function loadData() {
    const version = ++loadVersion;
    recordsController?.abort();
    recordsController = new AbortController();
    const { signal } = recordsController;
    updateNavigation();
    notice('dataStatus', '');
    rows = []; comparison = []; allRows = [];
    // A failed or empty reload must never leave exportable previous records.
    if (byId('historyRows')) byId('historyRows').replaceChildren();
    if (byId('comparisonStatus')) byId('comparisonStatus').textContent = '';
    if (byId('downloadCsv')) byId('downloadCsv').disabled = true;
    if (byId('useAverages')) byId('useAverages').disabled = true;
    if (page === 'predictor') renderMetrics();
    try {
      if (byId('fromYear') && Number(byId('fromYear').value) > Number(byId('toYear').value)) {
        throw new Error('Choose a start year that is earlier than or equal to the end year.');
      }
      setRecordsState('loading');
      if (page === 'history') {
        const selectedYears = Number(byId('toYear').value) - Number(byId('fromYear').value) + 1;
        const comparing = byId('compareDistrict').value && byId('compareDistrict').value !== district();
        const count = Math.min(10, selectedYears * (comparing ? 2 : 1));
        document.querySelectorAll('.table-skeleton tr').forEach((row, index) => { row.hidden = index >= count; });
      }
      byId('filterContext').textContent = 'Loading historical records…';
      if (byId('recordCount')) byId('recordCount').textContent = 'Loading records…';
      const compare = byId('compareDistrict')?.value;
      if (compare && compare !== district()) loadComparison(version, crop(), compare, signal);
      const data = await records(crop(), district(), signal);
      if (version !== loadVersion || signal.aborted) return;
      allRows = data; rows = inPeriod(allRows);
      setRecordsState(rows.length || comparison.length ? 'success' : 'empty');
      byId('filterContext').textContent = `${rows.length} records · ${rows.length ? `${rows[0].Year}–${rows.at(-1).Year}` : 'No matching years'}`;
      if (!rows.length) notice('dataStatus', 'No primary district records match your selection. Try a different crop, district, or year range.');
      if (!window.Chart) notice('dataStatus', 'Charts could not load. Refresh the page or use the historical data table.', true);
      if (page === 'overview') renderOverview();
      if (page === 'history') renderHistory();
      if (page === 'predictor') {
        byId('useAverages').disabled = !allRows.length;
        for (const field of fields) {
          const values = allRows.map(row => row[columns[field]]);
          byId(`${field}Observed`).textContent = values.length
            ? `Local historical range: ${format(Math.min(...values))}–${format(Math.max(...values))} ${units[columns[field]]}` : 'No historical range available';
        }
        if (firstLoad) { fillAverages(); firstLoad = false; }
      }
    } catch (error) {
      if (version !== loadVersion || signal.aborted) return;
      recordsController.abort();
      setRecordsState('error');
      if (byId('comparisonStatus')) byId('comparisonStatus').textContent = '';
      byId('filterContext').textContent = 'Historical records unavailable';
      notice('dataStatus', `${error.message} Retry historical data or change a filter.`, true);
      if (page === 'overview') {
        renderOverview();
        byId('summaryPeriod').textContent = 'Records unavailable';
        byId('overviewInsight').textContent = 'Historical insights could not be loaded. Retry historical data or change a filter.';
      }
      if (page === 'history') { byId('historyRows').replaceChildren(); byId('recordCount').textContent = 'Records unavailable'; }
      if (page === 'predictor') fields.forEach(field => { byId(`${field}Observed`).textContent = 'Historical range unavailable'; });
    }
  }

  async function loadComparison(version, selectedCrop, selectedDistrict, signal) {
    byId('comparisonStatus').textContent = `Loading ${selectedDistrict} comparison…`;
    try {
      const data = await records(selectedCrop, selectedDistrict, signal);
      if (version !== loadVersion || signal.aborted) return;
      comparison = inPeriod(data);
      byId('comparisonStatus').textContent = comparison.length
        ? `Comparing with ${selectedDistrict}` : `No matching records for ${selectedDistrict}.`;
      const state = document.querySelector('[data-records]').dataset.state;
      if (state === 'success' || state === 'empty') {
        setRecordsState(rows.length || comparison.length ? 'success' : 'empty');
        renderHistory();
      }
    } catch (error) {
      if (version !== loadVersion || signal.aborted) return;
      byId('comparisonStatus').textContent = `${selectedDistrict} comparison unavailable. Primary district records load independently.`;
    }
  }

  function renderMetrics() {
    if (!metrics || page !== 'predictor') return;
    const container = byId('modelMetrics');
    if (metricsCrop === crop() && container.dataset.state === 'success') return;
    container.replaceChildren();
    const description = document.createElement('p');
    description.textContent = `${metrics.model} · Evaluation on unseen years ${metrics.holdout_years.join(' and ')}. MAE is average absolute error; lower is better.`;
    container.append(description);
    const grid = document.createElement('div');
    grid.className = 'metric-grid';
    const perCrop = metrics.per_crop[crop()];
    for (const [label, value] of [['Overall MAE', `${format(metrics.holdout.mae, 3)} t/ha`],
      [`${crop()} MAE`, perCrop ? `${format(perCrop.mae, 3)} t/ha` : 'Unavailable'],
      ['Crop-mean baseline MAE', `${format(metrics.baseline_holdout.mae, 3)} t/ha`]]) {
      const cell = document.createElement('div'), strong = document.createElement('strong'), text = document.createElement('span');
      strong.textContent = value; text.textContent = label; cell.append(strong, text); grid.append(cell);
    }
    container.append(grid);
    for (const message of [
      `Overall R²: ${scoreText(metrics.holdout.r2)}; ${crop()} R²: ${scoreText(perCrop?.r2)}. Overall R² includes differences between crop types. Negative per-crop R² means the estimate is weaker than that crop’s test-set mean.`,
      'Evaluation was completed before refitting the serving model on all years. Historical error is not a confidence interval or a guarantee for a new scenario.',
      metrics.holdout.mae >= metrics.baseline_holdout.mae
        ? 'On the held-out years, this model did not outperform a simple historical crop-average baseline. More informative data is needed to establish weather-related predictive value.'
        : `Held-out MAE improvement over the crop-average baseline: ${format(metrics.baseline_holdout.mae - metrics.holdout.mae, 4)} t/ha. A small improvement does not establish strong weather-driven predictive value.`,
      'Dataset provenance is undocumented. Estimates outside historical weather ranges are less reliable.',
    ]) { const p = document.createElement('p'); p.textContent = message; container.append(p); }
    metricsCrop = crop();
  }

  async function loadModelInfo() {
    modelController?.abort();
    const controller = new AbortController();
    modelController = controller;
    const container = byId('modelMetrics');
    if (!modelSkeleton) modelSkeleton = container.querySelector('.model-skeleton').cloneNode(true);
    container.replaceChildren(modelSkeleton.cloneNode(true));
    setState(container, 'loading');
    try {
      const data = await requestJson('/model_info', { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (!data?.holdout || !data.per_crop) {
        showModelMessage('empty', 'No model evaluation has been published yet.');
        return;
      }
      metrics = window.FieldwiseClient.validateMetrics(data);
      renderMetrics();
      setState(container, 'success');
    } catch (error) {
      if (controller.signal.aborted) return;
      metrics = null;
      showModelMessage('error', error.message || 'Model evaluation could not be loaded.');
    } finally {
      if (modelController === controller) modelController = null;
    }
  }

  function showModelMessage(state, message) {
    const container = byId('modelMetrics');
    const content = document.createElement('div');
    content.className = 'model-message';
    const text = document.createElement('p');
    text.textContent = message;
    const retry = document.createElement('button');
    retry.type = 'button'; retry.className = 'button button-soft'; retry.textContent = 'Retry evaluation';
    retry.addEventListener('click', loadModelInfo);
    content.append(text, retry);
    container.replaceChildren(content);
    setState(container, state);
  }

  function finishPrediction(state = hasPrediction ? 'success' : 'idle') {
    byId('predictBtn').disabled = false;
    byId('predictBtn').textContent = 'Predict yield ↗';
    setState(byId('resultPanel'), state);
    setState(byId('comparisonPanel'), state);
    setState(byId('predictionChart').parentElement, state === 'success' && failedCharts.has('predictionChart') ? 'error' : state);
    if (state === 'idle') byId('comparisonPanel').hidden = true;
  }

  async function predict(event) {
    event.preventDefault();
    const form = byId('predictionForm');
    if (!form.reportValidity() || predictionController) return;
    const version = ++predictionVersion;
    const controller = new AbortController();
    predictionController = controller;
    let state = 'error';
    const payload = new URLSearchParams(new FormData(form));
    payload.set('crop', crop()); payload.set('district', district());
    byId('predictBtn').disabled = true;
    byId('predictBtn').textContent = 'Calculating estimate…';
    setState(byId('resultPanel'), 'loading');
    setState(byId('comparisonPanel'), 'loading');
    setState(byId('predictionChart').parentElement, 'loading');
    byId('comparisonPanel').hidden = false;
    byId('resultBadge').textContent = 'Calculating…';
    notice('predictionStatus', '');
    try {
      const data = await requestJson('/predict', { method: 'POST', body: payload, signal: controller.signal });
      if (version !== predictionVersion || controller.signal.aborted) return;
      window.FieldwiseClient.validatePrediction(data, crop(), district());
      byId('predictionEmpty').hidden = true; byId('predictionResult').hidden = false;
      byId('comparisonPanel').hidden = false; hasPrediction = true;
      byId('resultBadge').textContent = 'Estimate ready';
      byId('predictionContext').textContent = `${data.crop} · ${data.district}`;
      byId('predictedYield').textContent = format(data.predicted, 3);
      byId('selectedConditions').textContent = `Tested inputs: ${format(Number(payload.get('rainfall')))} mm rainfall, ${format(Number(payload.get('temperature')))} °C, ${format(Number(payload.get('humidity')))}% humidity.`;
      byId('yieldGap').textContent = data.gap > 0
        ? `Best tested scenario: ${format(data.optimal, 3)} t/ha · ${format(data.gap, 3)} t/ha higher`
        : 'Your selected conditions gave the highest yield among the tested scenarios.';
      byId('bestRain').textContent = `${format(data.best_rain)} mm`;
      byId('bestTemp').textContent = `${format(data.best_temp)} °C`;
      byId('bestHum').textContent = `${format(data.best_hum)} %`;
      for (const [id, key, field] of [['bestRain', 'best_rain', 'rainfall'], ['bestTemp', 'best_temp', 'temperature'], ['bestHum', 'best_hum', 'humidity']]) {
        const delta = data[key] - Number(payload.get(field));
        byId(`${id}Change`).textContent = Math.abs(delta) < 0.05 ? 'Same as your input'
          : `${delta > 0 ? '+' : '−'}${format(Math.abs(delta))} ${units[columns[field]]} vs your input`;
      }
      draw('predictionChart', 'bar', ['Your conditions', 'Best tested scenario'], [{
        label: 'Yield (t/ha)', data: [data.predicted, data.optimal], backgroundColor: ['#aebd83', '#386646'],
        borderRadius: 6, maxBarThickness: 65,
      }], { y: 'Yield (t/ha)' });
      notice('predictionStatus', data.warnings?.join('\n') || '');
      state = 'success';
    } catch (error) {
      if (version === predictionVersion && !controller.signal.aborted) {
        byId('resultBadge').textContent = 'Try again';
        notice('predictionStatus', error.message, true);
      }
    } finally {
      if (version === predictionVersion) {
        predictionController = null;
        finishPrediction(state);
      }
    }
  }

  function exportCsv() {
    const records = [...rows, ...comparison].sort((a, b) => a.Year - b.Year || a.District.localeCompare(b.District));
    if (!records.length) return;
    const csv = window.FieldwiseClient.historicalCsv(records);
    const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8;' }));
    const link = document.createElement('a'); link.href = url;
    link.download = `${crop()}-${district()}-${byId('fromYear').value}-${byId('toYear').value}.csv`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const query = new URLSearchParams(location.search);
  for (const [id, parameter] of [['crop', 'crop'], ['district', 'district'], ['fromYear', 'from'], ['toYear', 'to'],
    ['compareDistrict', 'compare'], ['historyMeasure', 'measure']]) {
    const element = byId(id), value = query.get(parameter);
    if (!element) continue;
    if (value && [...element.options].some(option => option.value === value)) {
      element.value = value;
      // Enhanced dropdowns initialize before URL filters are restored.
      element.dispatchEvent(new Event('change', { bubbles: true }));
    }
    element.addEventListener('change', () => {
      if (id === 'historyMeasure') { updateNavigation(); renderHistory(false); }
      else { markStale(); loadData(); }
    });
  }
  if (page === 'predictor') {
    fields.forEach(field => {
      const number = byId(`${field}Number`), range = byId(`${field}Range`);
      range.addEventListener('input', () => { number.value = range.value; markStale(); });
      number.addEventListener('input', () => { range.value = number.value; markStale(); });
    });
    byId('predictionForm').addEventListener('submit', predict);
    byId('useAverages').addEventListener('click', fillAverages);
    byId('resetInputs').addEventListener('click', () => { byId('predictionForm').reset(); markStale(); });
    loadModelInfo();
  }
  if (page === 'history') {
    for (const view of ['chart', 'table']) {
      byId(`${view}View`).addEventListener('click', () => {
        byId('historyChartView').hidden = view !== 'chart';
        byId('historyTableView').hidden = view !== 'table';
        for (const other of ['chart', 'table']) {
          byId(`${other}View`).classList.toggle('selected', view === other);
          byId(`${other}View`).setAttribute('aria-pressed', String(view === other));
        }
        if (view === 'chart') charts.get('historyChart')?.resize();
      });
    }
    byId('downloadCsv').addEventListener('click', exportCsv);
  }
  window.addEventListener('pagehide', () => {
    recordsController?.abort(); predictionController?.abort(); modelController?.abort();
  });
  byId('retryData')?.addEventListener('click', loadData);
  window.addEventListener('pageshow', event => {
    if (event.persisted) {
      if (predictionController) markStale();
      loadData();
      if (page === 'predictor' && !metrics) loadModelInfo();
    }
  });
  loadData();
})();
