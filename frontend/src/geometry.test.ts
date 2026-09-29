import assert from 'node:assert/strict';
import test from 'node:test';

import { measureManualAxis, pointFromClient } from './geometry.ts';

test('vertical axis has zero angle', () => {
  assert.deepEqual(measureManualAxis([50, 10], [50, 110], 200, 200), { angleDeg: 0, violation: false });
});

test('exactly five degrees is not a violation', () => {
  const dx = 100 * Math.tan(5 * Math.PI / 180);
  const measured = measureManualAxis([50 + dx, 10], [50, 110], 200, 200);
  assert.ok(Math.abs(measured.angleDeg - 5) < 1e-10);
  assert.equal(measured.violation, false);
});

test('an angle above five degrees is a violation', () => {
  const measured = measureManualAxis([62, 10], [50, 110], 200, 200);
  assert.ok(measured.angleDeg > 5);
  assert.equal(measured.violation, true);
});

test('rejects inverted, nonfinite and out-of-frame points', () => {
  assert.throws(() => measureManualAxis([50, 110], [50, 10], 200, 200));
  assert.throws(() => measureManualAxis([NaN, 10], [50, 110], 200, 200));
  assert.throws(() => measureManualAxis([-1, 10], [50, 110], 200, 200));
});

test('rejects invalid native frame dimensions', () => {
  assert.throws(() => pointFromClient(10, 10, { left: 0, top: 0, width: 0, height: 100 }, 300, 150));
});

test('maps pointer to a letterboxed non-square image', () => {
  const bounds = { left: 0, top: 0, width: 600, height: 600 };
  assert.deepEqual(pointFromClient(300, 150, bounds, 300, 150), [150, 0]);
  assert.deepEqual(pointFromClient(300, 450, bounds, 300, 150), [150, 150]);
  assert.deepEqual(pointFromClient(600, 300, bounds, 300, 150), [300, 75]);
});
