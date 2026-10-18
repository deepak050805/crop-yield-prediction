(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const page = document.body.dataset.page;
  const charts = new Map();
  const fields = ['rainfall', 'temperature', 'humidity'];
  const units = { Yield: 't/ha', Rainfall: 'mm', Temperature: '°C', Humidity: '%' };
  const columns = { rainfall: 'Rainfall', temperature: 'Temperature', humidity: 'Humidity' };
  let rows = [], comparison = [], allRows = [], metrics = null;
  let loadVersion = 0, predictionVersion = 0, hasPrediction = false, firstLoad = true;
  const crop = () => byId('crop').value;
  const district = () => byId('district').value;
  const format = (number, digits = 1) => Number(number).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
  const mean = (data, key) => data.reduce((sum, row) => sum + Number(row[key]), 0) / data.length;

  if (window.Chart) {
    Chart.defaults.font.family = "'DM Sans', sans-serif";
    Chart.defaults.font.size = 10;
    Chart.defaults.color = '#819078';
    Chart.defaults.animation = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : { duration: 450 };
  }

  function notice(id, message, error = false) {
    const element = byId(id);
    if (!element) return;
    element.textContent = message;
    element.hidden = !message;
    element.classList.toggle('error', error);
  }

  function draw(id, type, labels, datasets, axisTitles = {}) {
    if (!byId(id) || !window.Chart) return;
    if (charts.has(id)) charts.get(id).destroy();
    const chart = new Chart(byId(id), {
      type,
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: type === 'scatter' ? 'nearest' : 'index', intersect: false },
        plugins: {
          legend: { display: id === 'historyChart', position: 'bottom', labels: { usePointStyle: true, boxWidth: 7, padding: 22 } },
          tooltip: { backgroundColor: '#264c36', padding: 12, cornerRadius: 8 },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { maxTicksLimit: 8 },
            title: { display: Boolean(axisTitles.x), text: axisTitles.x } },
          y: { grid: { color: '#edf0e7' }, border: { display: false },
            beginAtZero: type === 'bar', ticks: { maxTicksLimit: 5 },
            title: { display: Boolean(axisTitles.y), text: axisTitles.y } },
        },
      },
    });
    charts.set(id, chart);
  }

  function line(label, values, color = '#537b4d', fill = false) {
    return { label, data: values, borderColor: color, backgroundColor: fill ? '#dce8cf70' : color,
      fill, borderWidth: 2, tension: 0.25, pointRadius: 3, pointHoverRadius: 6,
      pointBackgroundColor: '#fff', pointBorderWidth: 2 };
  }

  async function records(selectedCrop, selectedDistrict) {
    const response = await fetch('/get_data', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ crop: selectedCrop, district: selectedDistrict }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Historical data could not be loaded.');
    return data;
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
    history.replaceState(null, '', `${location.pathname}?${query}`);
    document.querySelectorAll('[data-nav]').forEach(link => {
      const url = new URL(link.href);
      url.search = query.toString();
      link.href = url.toString();
    });
  }

  function markStale() {
    predictionVersion += 1;
    if (!hasPrediction && !byId('predictBtn')?.disabled) return;
    byId('resultBadge').textContent = 'Inputs changed';
    notice('predictionStatus', 'Your inputs have changed. Select Predict yield to update this estimate.');
  }

  function renderOverview() {
    for (const [id, key, digits] of [['avgYield', 'Yield', 2], ['avgRain', 'Rainfall', 0],
      ['avgTemp', 'Temperature', 1], ['avgHum', 'Humidity', 0]]) {
      byId(id).textContent = rows.length ? format(mean(rows, key), digits) : '—';
    }
    byId('summaryPeriod').textContent = rows.length ? `${rows[0].Year}–${rows.at(-1).Year} · ${rows.length} records` : 'No matching records';
    draw('yieldChart', 'line', rows.map(row => row.Year), [line('Yield (t/ha)', rows.map(row => row.Yield), '#537b4d', true)]);
    draw('rainChart', 'scatter', [], [{ label: 'Rainfall vs yield',
      data: rows.map(row => ({ x: row.Rainfall, y: row.Yield })), backgroundColor: '#759251', pointRadius: 5, pointHoverRadius: 7 }],
      { x: 'Rainfall (mm)', y: 'Yield (t/ha)' });
    byId('insightTitle').textContent = `${crop()} in ${district()}`;
    if (!rows.length) {
      byId('overviewInsight').textContent = 'Choose another year range to explore historical performance.';
      return;
    }
    const best = rows.reduce((previous, current) => current.Yield > previous.Yield ? current : previous);
    byId('overviewInsight').textContent = `Highest recorded yield in this selection: ${format(best.Yield, 2)} t/ha in ${best.Year}. Average across ${rows.length} seasons: ${format(mean(rows, 'Yield'), 2)} t/ha.`;
  }

  function renderHistory() {
    const measure = byId('historyMeasure').value;
    const datasets = [line(district(), rows.map(row => ({ x: row.Year, y: row[measure] })))];
    if (comparison.length) datasets.push(line(byId('compareDistrict').value, comparison.map(row => ({ x: row.Year, y: row[measure] })), '#ba9654'));
    const years = [...new Set([...rows, ...comparison].map(row => row.Year))].sort((a, b) => a - b);
    draw('historyChart', 'line', years, datasets, { y: `${measure} (${units[measure]})` });
    byId('historyTitle').textContent = `${measure} through the years`;
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
    updateNavigation();
    byId('filterContext').textContent = 'Loading historical records…';
    notice('dataStatus', '');
    rows = []; comparison = []; allRows = [];
    if (page === 'overview') renderOverview();
    if (page === 'history') renderHistory();
    if (byId('useAverages')) byId('useAverages').disabled = true;
    try {
      if (byId('fromYear') && Number(byId('fromYear').value) > Number(byId('toYear').value)) {
        throw new Error('Choose a start year that is earlier than or equal to the end year.');
      }
      const compare = byId('compareDistrict')?.value;
      const data = await Promise.all([records(crop(), district()),
        compare && compare !== district() ? records(crop(), compare) : Promise.resolve([])]);
      if (version !== loadVersion) return;
      allRows = data[0]; rows = inPeriod(allRows); comparison = inPeriod(data[1]);
      byId('filterContext').textContent = `${rows.length} records · ${rows.length ? `${rows[0].Year}–${rows.at(-1).Year}` : 'No matching years'}`;
      if (!rows.length) notice('dataStatus', 'No historical records match your selection. Try a different crop, district, or year range.');
      if (!window.Chart) notice('dataStatus', 'Charts could not load. Refresh the page or use the historical data table.', true);
      if (page === 'overview') renderOverview();
      if (page === 'history') renderHistory();
      if (page === 'predictor') {
        byId('useAverages').disabled = !allRows.length;
        for (const field of fields) {
          const values = allRows.map(row => row[columns[field]]);
          byId(`${field}Observed`).textContent = values.length
            ? `Local historical range: ${format(Math.min(...values))}–${format(Math.max(...values))} ${units[columns[field]]}` : '';
        }
        if (firstLoad) { fillAverages(); firstLoad = false; }
        renderMetrics();
      }
    } catch (error) {
      if (version !== loadVersion) return;
      byId('filterContext').textContent = 'Historical records unavailable';
      notice('dataStatus', `${error.message} Change a filter to try again.`, true);
    }
  }

  function renderMetrics() {
    if (!metrics || page !== 'predictor') return;
    const container = byId('modelMetrics');
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
      `Overall R²: ${format(metrics.holdout.r2, 3)}; ${crop()} R²: ${perCrop ? format(perCrop.r2, 3) : 'unavailable'}. Overall R² includes differences between crop types. Negative per-crop R² means the estimate is weaker than that crop’s test-set mean.`,
      'Evaluation was completed before refitting the serving model on all years. Historical error is not a confidence interval or a guarantee for a new scenario.',
      metrics.holdout.mae >= metrics.baseline_holdout.mae
        ? 'On the held-out years, this model did not outperform a simple historical crop-average baseline. More informative data is needed to establish weather-related predictive value.'
        : `Held-out MAE improvement over the crop-average baseline: ${format(metrics.baseline_holdout.mae - metrics.holdout.mae, 4)} t/ha. A small improvement does not establish strong weather-driven predictive value.`,
      'Dataset provenance is undocumented. Estimates outside historical weather ranges are less reliable.',
    ]) { const p = document.createElement('p'); p.textContent = message; container.append(p); }
  }

  async function predict(event) {
    event.preventDefault();
    const form = byId('predictionForm');
    if (!form.reportValidity()) return;
    const version = ++predictionVersion;
    const payload = new URLSearchParams(new FormData(form));
    payload.set('crop', crop()); payload.set('district', district());
    byId('predictBtn').disabled = true;
    byId('predictBtn').textContent = 'Calculating estimate…';
    byId('resultPanel').setAttribute('aria-busy', 'true');
    notice('predictionStatus', '');
    try {
      const response = await fetch('/predict', { method: 'POST', body: payload });
      const data = await response.json();
      if (version !== predictionVersion) return;
      if (!response.ok) throw new Error(data.error || 'Prediction failed. Please try again.');
      byId('predictionEmpty').hidden = true; byId('predictionResult').hidden = false;
      byId('comparisonPanel').hidden = false; hasPrediction = true;
      byId('resultBadge').textContent = 'Estimate ready';
      byId('predictionContext').textContent = `${data.crop} · ${data.district}`;
      byId('predictedYield').textContent = format(data.predicted, 3);
      byId('yieldGap').textContent = data.gap > 0
        ? `Best tested scenario: ${format(data.optimal, 3)} t/ha · ${format(data.gap, 3)} t/ha higher`
        : 'Your selected conditions gave the highest yield among the tested scenarios.';
      byId('bestRain').textContent = `${format(data.best_rain)} mm`;
      byId('bestTemp').textContent = `${format(data.best_temp)} °C`;
      byId('bestHum').textContent = `${format(data.best_hum)} %`;
      draw('predictionChart', 'bar', ['Your conditions', 'Best tested scenario'], [{
        label: 'Yield (t/ha)', data: [data.predicted, data.optimal], backgroundColor: ['#aebd83', '#386646'],
        borderRadius: 6, maxBarThickness: 65,
      }], { y: 'Yield (t/ha)' });
      notice('predictionStatus', data.warnings.join('\n'));
    } catch (error) {
      if (version === predictionVersion) {
        byId('resultBadge').textContent = hasPrediction ? 'Previous estimate' : 'Try again';
        notice('predictionStatus', error.message, true);
      }
    } finally {
      byId('predictBtn').disabled = false;
      byId('predictBtn').textContent = 'Predict yield ↗';
      byId('resultPanel').setAttribute('aria-busy', 'false');
    }
  }

  function exportCsv() {
    const records = [...rows, ...comparison].sort((a, b) => a.Year - b.Year || a.District.localeCompare(b.District));
    if (!records.length) return;
    const headers = ['District', 'Crop', 'Year', 'Temperature', 'Rainfall', 'Humidity', 'Yield'];
    const quote = value => `"${String(value).replaceAll('"', '""')}"`;
    const csv = [headers, ...records.map(row => headers.map(key => row[key]))].map(row => row.map(quote).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8;' }));
    const link = document.createElement('a'); link.href = url;
    link.download = `${crop()}-${district()}-${byId('fromYear').value}-${byId('toYear').value}.csv`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const query = new URLSearchParams(location.search);
  for (const [id, parameter] of [['crop', 'crop'], ['district', 'district'], ['fromYear', 'from'], ['toYear', 'to']]) {
    const element = byId(id), value = query.get(parameter);
    if (!element) continue;
    if (value && [...element.options].some(option => option.value === value)) element.value = value;
    element.addEventListener('change', () => { markStale(); loadData(); });
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
    fetch('/model_info').then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      metrics = data; renderMetrics();
    }).catch(error => { byId('modelMetrics').textContent = error.message || 'Model evaluation could not be loaded.'; });
  }
  if (page === 'history') {
    byId('compareDistrict').addEventListener('change', loadData);
    byId('historyMeasure').addEventListener('change', renderHistory);
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
  loadData();
})();
