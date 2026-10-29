const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { JSDOM } = require('jsdom');
const script = readFileSync(require.resolve('../static/js/theme.js'), 'utf8');

function setup(t, { saved = null, dark = false, blocked = false } = {}) {
  const dom = new JSDOM('<button id="themeToggle"><span id="themeLabel"></span></button>', {
    url: 'http://localhost', runScripts: 'outside-only',
  });
  t.after(() => dom.window.close());
  const { window } = dom;
  let changed;
  const media = { matches: dark, addEventListener: (event, listener) => { changed = listener; } };
  window.matchMedia = () => media;
  if (saved !== null) window.localStorage.setItem('fieldwise-theme', saved);
  if (blocked) Object.defineProperty(window, 'localStorage', { get() { throw new Error('Storage blocked'); } });
  window.eval(script);
  window.document.dispatchEvent(new window.Event('DOMContentLoaded'));
  return { window, theme: () => window.document.documentElement.dataset.theme,
    button: window.document.getElementById('themeToggle'),
    system(dark) { media.matches = dark; changed(); } };
}

test('saved themes override system preference and toggles persist an accessible state', t => {
  const c = setup(t, { saved: 'light', dark: true });
  assert.equal(c.theme(), 'light');
  c.button.click();
  assert.equal(c.theme(), 'dark');
  assert.equal(c.window.localStorage.getItem('fieldwise-theme'), 'dark');
  assert.equal(c.button.getAttribute('aria-pressed'), 'true');
  assert.equal(c.button.getAttribute('aria-label'), 'Switch to light mode');
  c.system(false);
  assert.equal(c.theme(), 'dark', 'An explicit preference wins over later system changes');
});

test('invalid preferences follow live system changes and unavailable storage remains usable', t => {
  const c = setup(t, { saved: 'invalid' });
  c.system(true);
  assert.equal(c.theme(), 'dark');
  const blocked = setup(t, { blocked: true, dark: true });
  assert.equal(blocked.theme(), 'dark');
  assert.doesNotThrow(() => blocked.button.click());
  assert.equal(blocked.theme(), 'light');
});

test('theme changes in other tabs synchronize and clearing storage restores the system choice', t => {
  const c = setup(t, { saved: 'light' });
  c.window.dispatchEvent(new c.window.StorageEvent('storage', { key: 'fieldwise-theme', newValue: 'dark' }));
  assert.equal(c.theme(), 'dark');
  c.window.dispatchEvent(new c.window.StorageEvent('storage', { key: null }));
  assert.equal(c.theme(), 'light');
  c.system(true);
  assert.equal(c.theme(), 'dark');
});
