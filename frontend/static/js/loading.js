(() => {
  'use strict';
  function setState(element, state) {
    if (!element) return;
    element.dataset.state = state;
    element.setAttribute('aria-busy', String(state === 'loading'));
    element.querySelectorAll('.loading-skeleton').forEach(placeholder => {
      // An independently loaded child (for example a cached image) owns its state.
      if (placeholder.closest('[data-state]') === element) placeholder.hidden = state !== 'loading';
    });
  }
  window.FieldwiseLoading = { setState };
  document.querySelectorAll('[data-state]').forEach(element => setState(element, element.dataset.state));

  document.querySelectorAll('[data-image]').forEach(frame => {
    const image = frame.querySelector('img');
    if (!image) return;
    const loaded = () => setState(frame, 'success');
    const failed = () => setState(frame, 'error');
    image.addEventListener('load', loaded, { once: true });
    image.addEventListener('error', failed, { once: true });
    if (image.complete) setState(frame, image.naturalWidth > 0 ? 'success' : 'error');
  });
})();
