import assert from 'node:assert/strict';
import test from 'node:test';
import { zoomAt } from './viewTransform.ts';

test('zoom keeps image point under cursor fixed', () => {
  assert.deepEqual(zoomAt({ zoom: 1, x: 0, y: 0 }, 2, { x: 100, y: -40 }), { zoom: 2, x: -100, y: 40 });
});
test('zoom transforms existing pan offset', () => {
  assert.deepEqual(zoomAt({ zoom: 2, x: 20, y: -10 }, 1, { x: 100, y: 50 }), { zoom: 1, x: 60, y: 20 });
});
test('zoom clamps scale before calculating offset', () => {
  assert.deepEqual(zoomAt({ zoom: 1, x: 0, y: 0 }, 100, { x: 10, y: 0 }), { zoom: 8, x: -70, y: 0 });
  assert.deepEqual(zoomAt({ zoom: 1, x: 0, y: 0 }, 0.01, { x: 100, y: 0 }), { zoom: 0.25, x: 75, y: 0 });
});
