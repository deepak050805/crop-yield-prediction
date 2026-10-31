(() => {
  'use strict';
  let opened = null;

  document.querySelectorAll('.themed-select > select').forEach(select => {
    const wrapper = select.parentElement;
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'select-trigger';
    trigger.id = `${select.id}Trigger`;
    trigger.setAttribute('role', 'combobox');
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', `${select.id}Listbox`);
    trigger.setAttribute('aria-autocomplete', 'none');
    const label = select.labels[0];
    trigger.setAttribute('aria-label', label
      ? Array.from(label.childNodes).filter(node => node.nodeType === Node.TEXT_NODE)
        .map(node => node.textContent).join(' ').trim()
      : select.getAttribute('aria-label') || select.id);
    const value = document.createElement('span');
    value.className = 'select-value';
    trigger.append(value);
    const menu = document.createElement('span');
    menu.id = `${select.id}Listbox`;
    menu.className = 'select-menu';
    menu.setAttribute('role', 'listbox');
    menu.setAttribute('aria-label', trigger.getAttribute('aria-label'));
    menu.hidden = true;
    let activeIndex = select.selectedIndex;
    let search = '', lastKeyTime = 0;
    const hidden = index => select.options[index]?.hidden || select.options[index]?.closest('optgroup')?.hidden;
    const disabled = index => !select.options[index] || select.options[index].disabled ||
      select.options[index].parentElement.disabled || hidden(index);
    let options = [];
    function rebuildOptions() {
      options = Array.from(select.options).map((option, index) => {
        const item = document.createElement('span');
        item.id = `${select.id}Option${index}`;
        item.className = 'select-option';
        item.setAttribute('role', 'option');
        item.textContent = option.label;
        item.title = option.label;
        if (disabled(index)) item.setAttribute('aria-disabled', 'true');
        item.addEventListener('pointermove', event => {
          if (event.pointerType !== 'touch' && !disabled(index)) activate(index, false);
        });
        item.addEventListener('click', event => {
          event.preventDefault();
          if (!disabled(index)) choose(index);
        });
        return item;
      });
      menu.replaceChildren(...options);
      activeIndex = select.selectedIndex;
    }
    rebuildOptions();
    const enabled = () => options.map((_, index) => index).filter(index => !disabled(index));

    function sync() {
      value.textContent = select.options[select.selectedIndex]?.label || '';
      trigger.title = value.textContent;
      trigger.disabled = select.matches(':disabled');
      for (const attribute of ['aria-describedby', 'aria-invalid']) {
        if (select.hasAttribute(attribute)) trigger.setAttribute(attribute, select.getAttribute(attribute));
        else trigger.removeAttribute(attribute);
      }
      options.forEach((option, index) => {
        option.hidden = Boolean(hidden(index));
        option.setAttribute('aria-selected', String(index === select.selectedIndex));
        option.setAttribute('aria-disabled', String(Boolean(disabled(index))));
      });
      if (trigger.disabled) close();
      else if (!menu.hidden) activate(disabled(activeIndex) ? enabled()[0] : activeIndex, false);
    }

    function activate(index, scroll = true) {
      activeIndex = index;
      options.forEach((option, itemIndex) => option.classList.toggle('is-active', itemIndex === index));
      const option = options[index];
      if (!option) { trigger.removeAttribute('aria-activedescendant'); return; }
      trigger.setAttribute('aria-activedescendant', option.id);
      if (scroll) {
        const top = option.offsetTop, bottom = top + option.offsetHeight;
        if (top < menu.scrollTop) menu.scrollTop = Math.max(0, top - 4);
        else if (bottom > menu.scrollTop + menu.clientHeight) menu.scrollTop = bottom - menu.clientHeight + 4;
      }
    }

    function close() {
      menu.hidden = true;
      wrapper.classList.remove('is-open');
      trigger.setAttribute('aria-expanded', 'false');
      trigger.removeAttribute('aria-activedescendant');
      search = '';
      if (opened?.wrapper === wrapper) opened = null;
    }

    function open() {
      if (trigger.disabled) return;
      opened?.close();
      menu.hidden = false;
      const rect = trigger.getBoundingClientRect();
      const below = window.innerHeight - rect.bottom - 12, above = rect.top - 12;
      const upwards = below < 180 && above > below;
      wrapper.classList.toggle('opens-up', upwards);
      menu.style.maxHeight = `${Math.min(216, Math.max(40, upwards ? above : below))}px`;
      wrapper.classList.add('is-open');
      trigger.setAttribute('aria-expanded', 'true');
      opened = { wrapper, close };
      activate(disabled(select.selectedIndex) ? enabled()[0] : select.selectedIndex);
    }

    function choose(index) {
      if (trigger.disabled || disabled(index)) return;
      const changed = select.selectedIndex !== index;
      select.selectedIndex = index;
      sync();
      close();
      trigger.focus({ preventScroll: true });
      if (changed) select.dispatchEvent(new Event('change', { bubbles: true }));
    }

    trigger.addEventListener('click', event => {
      event.preventDefault();
      menu.hidden ? open() : close();
    });
    // Keep focus on the combobox while its popup is used with a pointer.
    menu.addEventListener('pointerdown', event => event.preventDefault());
    trigger.addEventListener('keydown', event => {
      const key = event.key;
      if (key === 'Tab') { close(); return; }
      if (key === 'Escape') { event.preventDefault(); close(); return; }
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(key)) {
        event.preventDefault();
        const wasClosed = menu.hidden;
        if (wasClosed) open();
        const indices = enabled();
        let position = indices.indexOf(activeIndex);
        if (key === 'Home') position = 0;
        else if (key === 'End') position = indices.length - 1;
        else if (!wasClosed) position = Math.max(0, Math.min(indices.length - 1, position + (key === 'ArrowDown' ? 1 : -1)));
        activate(indices[Math.max(0, position)]);
        return;
      }
      if (key === 'Enter' || key === ' ') {
        event.preventDefault();
        if (menu.hidden) open();
        else if (activeIndex >= 0) choose(activeIndex);
        return;
      }
      if (key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        if (menu.hidden) open();
        const now = Date.now();
        search = now - lastKeyTime < 700 ? search + key.toLowerCase() : key.toLowerCase();
        lastKeyTime = now;
        const repeated = [...search].every(character => character === search[0]);
        const prefix = repeated ? key.toLowerCase() : search;
        const indices = enabled();
        const start = repeated ? indices.indexOf(activeIndex) + 1 : 0;
        const ordered = indices.slice(start).concat(indices.slice(0, start));
        const match = ordered.find(index => select.options[index].label.toLowerCase().startsWith(prefix));
        if (match !== undefined) activate(match);
      }
    });
    wrapper.addEventListener('focusout', event => {
      if (!wrapper.contains(event.relatedTarget)) close();
    });
    select.addEventListener('change', sync);
    select.addEventListener('input', sync);
    select.form?.addEventListener('reset', () => { close(); setTimeout(sync, 0); });
    wrapper.append(trigger, menu);
    sync();
    select.hidden = true;
    if (label) label.htmlFor = trigger.id;
    const observer = new MutationObserver(mutations => {
      if (mutations.some(mutation => mutation.type !== 'attributes' ||
        ['selected', 'label'].includes(mutation.attributeName))) rebuildOptions();
      sync();
    });
    observer.observe(select, { attributes: true, childList: true, characterData: true, subtree: true,
      attributeFilter: ['disabled', 'hidden', 'aria-describedby', 'aria-invalid', 'selected', 'label'] });
    if (select.closest('fieldset')) observer.observe(select.closest('fieldset'), { attributes: true, attributeFilter: ['disabled'] });
    window.addEventListener('pagehide', () => close());
  });

  document.addEventListener('pointerdown', event => {
    if (opened && !opened.wrapper.contains(event.target)) opened.close();
  });
  window.addEventListener('resize', () => opened?.close());
  document.addEventListener('scroll', event => {
    if (opened && !opened.wrapper.contains(event.target)) opened.close();
  }, true);
})();
