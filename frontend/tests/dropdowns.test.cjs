const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { JSDOM } = require('jsdom');
const script = readFileSync(require.resolve('../static/js/dropdowns.js'), 'utf8');

function setup(t, options = '<option>Wheat</option><option disabled>Barley</option><option>Rice</option><option>Rye</option>') {
  const dom = new JSDOM('<form><fieldset><label>Crop<span class="themed-select">' +
    '<select id="crop" aria-describedby="hint">' + options + '</select></span></label></fieldset></form>' +
    '<p id="hint">Choose a crop.</p><button id="outside">Outside</button>', { runScripts: 'outside-only' });
  t.after(() => dom.window.close());
  dom.window.eval(script);
  const get = id => dom.window.document.getElementById(id);
  return { window: dom.window, get,
    key: key => get('cropTrigger').dispatchEvent(new dom.window.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true })) };
}

test('combobox keyboard selection skips disabled options and commits one change', t => {
  const c = setup(t);
  let changes = 0;
  c.get('crop').addEventListener('change', () => changes++);
  c.get('cropTrigger').focus();
  c.key('ArrowDown'); c.key('ArrowDown');
  assert.equal(c.get('cropTrigger').getAttribute('aria-activedescendant'), 'cropOption2');
  assert.equal(c.get('crop').value, 'Wheat', 'Navigation alone must not commit a selection');
  c.key('Enter');
  assert.equal(c.get('crop').value, 'Rice');
  assert.equal(c.get('cropTrigger').textContent, 'Rice');
  assert.equal(changes, 1);
  assert.equal(c.get('cropListbox').hidden, true);
  assert.equal(c.window.document.activeElement, c.get('cropTrigger'));
});

test('typeahead cycles matching options and Escape preserves the selected value', t => {
  const c = setup(t);
  c.key('r'); c.key('r');
  assert.equal(c.get('cropTrigger').getAttribute('aria-activedescendant'), 'cropOption3');
  c.key('Escape');
  assert.equal(c.get('crop').value, 'Wheat');
  assert.equal(c.get('cropTrigger').hasAttribute('aria-activedescendant'), false);
});

test('dynamic disabled controls close their popup and keep accessible hints', async t => {
  const c = setup(t);
  c.get('cropTrigger').click();
  c.get('crop').disabled = true;
  await Promise.resolve();
  assert.equal(c.get('cropTrigger').disabled, true);
  assert.equal(c.get('cropListbox').hidden, true);
  assert.equal(c.get('cropTrigger').getAttribute('aria-describedby'), 'hint');
  c.get('crop').disabled = false;
  c.window.document.querySelector('fieldset').disabled = true;
  await Promise.resolve();
  assert.equal(c.get('cropTrigger').disabled, true);
});

test('form reset restores the visible selection and closes the popup', async t => {
  const c = setup(t);
  c.key('End'); c.key('Enter');
  c.get('cropTrigger').click();
  c.window.document.querySelector('form').reset();
  assert.equal(c.get('crop').value, 'Wheat');
  await new Promise(resolve => c.window.setTimeout(resolve, 0));
  assert.equal(c.get('cropTrigger').textContent, 'Wheat');
  assert.equal(c.get('cropListbox').hidden, true);
});

test('disabled option groups and options changed while open cannot be selected', async t => {
  const c = setup(t, '<option>Wheat</option><optgroup label="Unavailable" disabled><option>Rice</option><option>Rye</option></optgroup>');
  c.key('End'); c.key('Enter');
  assert.equal(c.get('crop').value, 'Wheat');
  assert.equal(c.get('cropOption1').getAttribute('aria-disabled'), 'true');
  c.window.document.querySelector('optgroup').disabled = false;
  await Promise.resolve();
  c.key('End');
  assert.equal(c.get('cropTrigger').getAttribute('aria-activedescendant'), 'cropOption2');
  c.get('crop').options[2].disabled = true;
  await Promise.resolve();
  c.key('Enter');
  assert.equal(c.get('crop').value, 'Wheat');
  c.get('crop').options[0].disabled = true;
  c.get('crop').options[1].disabled = true;
  await Promise.resolve();
  c.key('ArrowDown'); c.key('Enter');
  assert.equal(c.get('cropTrigger').hasAttribute('aria-activedescendant'), false);
  assert.equal(c.get('crop').value, 'Wheat');
});
