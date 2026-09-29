import assert from 'node:assert/strict';
import test from 'node:test';
import { intersectingItems, panelWidth, estimatedProgress } from './workspaceInteractions.ts';

test('marquee selects intersected rows in either drag direction', () => {
  const rows = [{ id: 'a', left: 0, top: 0, right: 200, bottom: 70 }, { id: 'b', left: 0, top: 75, right: 200, bottom: 145 }, { id: 'c', left: 0, top: 150, right: 200, bottom: 220 }];
  assert.deepEqual(intersectingItems(rows, { x: 180, y: 140 }, { x: 10, y: 15 }), ['a', 'b']);
});
test('panel resizing clamps widths to retain central viewer', () => {
  assert.equal(panelWidth(900, 'left', 1200, 320), 480);
  assert.equal(panelWidth(100, 'left', 1200, 320), 220);
  assert.equal(panelWidth(800, 'right', 1000, 260), 340);
});
test('estimated progress quickly reaches near completion but never claims 100%', () => {
  assert.equal(estimatedProgress(0), 15);
  assert.ok(estimatedProgress(3000) >= 70);
  assert.ok(estimatedProgress(6000) >= 90);
  assert.equal(estimatedProgress(60000), 96);
});
