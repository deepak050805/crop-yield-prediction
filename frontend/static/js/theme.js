(() => {
  'use strict';
  const key = 'fieldwise-theme';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  const valid = value => value === 'dark' || value === 'light';
  let preference = null;
  try {
    const saved = localStorage.getItem(key);
    if (valid(saved)) preference = saved;
  } catch { /* Use the system theme if storage is unavailable. */ }

  function applyTheme() {
    const theme = preference || (systemTheme.matches ? 'dark' : 'light');
    document.documentElement.dataset.theme = theme;
    const button = document.getElementById('themeToggle');
    if (button) {
      const dark = theme === 'dark';
      button.setAttribute('aria-pressed', String(dark));
      button.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} mode`);
      button.title = button.getAttribute('aria-label');
      document.getElementById('themeLabel').textContent = dark ? 'Dark mode' : 'Light mode';
    }
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
  }

  // Set the saved theme before styles and page content are painted.
  applyTheme();
  function initializeToggle() {
    document.getElementById('themeToggle')?.addEventListener('click', () => {
      preference = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(key, preference); } catch { /* Keep the choice for this page. */ }
      applyTheme();
    });
    applyTheme();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeToggle, { once: true });
  } else initializeToggle();
  systemTheme.addEventListener?.('change', () => { if (!preference) applyTheme(); });
  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) {
      preference = valid(event.newValue) ? event.newValue : null;
      applyTheme();
    }
  });
})();
