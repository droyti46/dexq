import assert from 'node:assert/strict';
import test from 'node:test';
import { commitAxis, undoAxis, redoAxis } from './axisHistory.ts';
import type { AxisHistory } from './axisHistory.ts';
const original: AxisHistory = { past: [], present: null, future: [] };
const axis = { top: [40, 10] as [number, number], bottom: [50, 150] as [number, number] };

test('one completed drag creates a single undo entry and redo restores it', () => {
  const committed = commitAxis({ ...original, present: axis }, null);
  assert.deepEqual(committed.past, [null]);
  const undone = undoAxis(committed);
  assert.equal(undone.present, null);
  assert.deepEqual(undone.future, [axis]);
  assert.deepEqual(redoAxis(undone), committed);
});
test('no motion creates no undo entry', () => {
  assert.deepEqual(commitAxis(original, null), original);
});
test('new edit after undo clears redo branch', () => {
  const undone = undoAxis(commitAxis({ ...original, present: axis }, null));
  const other = { ...axis, top: [45, 10] as [number, number] };
  assert.deepEqual(commitAxis({ ...undone, present: other }, null).future, []);
});
test('undo and redo without history preserve current axis', () => {
  assert.deepEqual(undoAxis(original), original);
  assert.deepEqual(redoAxis(original), original);
});
