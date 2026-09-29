import assert from 'node:assert/strict';
import test from 'node:test';
import { appendFiles, removeItems, updateItem, selectItems, reportCsv } from './projects.ts';
import type { WorkspaceItem } from './projects.ts';

const file = (name = 'image.png') => new File(['pixels'], name, { type: 'image/png' });
const item = (id: string): WorkspaceItem => ({ id, filename: `${id}.png`, size: 6, position: 1, status: 'queued', progress: 0 });

test('additional uploads retain old items and give duplicate filenames unique IDs', () => {
  const old = item('old');
  const next = appendFiles([old], [file(), file()]);
  assert.equal(next[0], old);
  assert.equal(next.length, 3);
  assert.equal(new Set(next.map((i) => i.id)).size, 3);
  assert.deepEqual(next.map((i) => i.position), [1, 2, 3]);
});

test('invalid upload and total capacity violations do not alter existing items', () => {
  const old = [item('old')];
  assert.throws(() => appendFiles(old, [file('bad.txt')]));
  assert.throws(() => appendFiles(old, [file('archive.zip'), file()]));
  assert.equal(appendFiles(Array.from({ length: 999 }, (_, i) => item(String(i))), [file()]).length, 1000);
  assert.throws(() => appendFiles(Array.from({ length: 1000 }, (_, i) => item(String(i))), [file()]));
  assert.equal(old.length, 1);
});

test('late results cannot restore a removed image', () => {
  const remaining = removeItems([item('a'), item('b')], ['a']);
  const next = updateItem(remaining, 'a', { status: 'ready', progress: 100 });
  assert.deepEqual(next.map((i) => i.id), ['b']);
});

test('removing archive also removes its streamed children but not unrelated images', () => {
  const next = removeItems([item('archive'), { ...item('child'), archiveId: 'archive' }, item('other')], ['archive']);
  assert.deepEqual(next.map((i) => i.id), ['other']);
});

test('plain selection, toggle and range selection remain independent of active image', () => {
  const ids = ['a', 'b', 'c', 'd'];
  assert.deepEqual(selectItems(ids, ['a'], 'c', 'a', 'range'), ['a', 'b', 'c']);
  assert.deepEqual(selectItems(ids, ['a', 'c'], 'c', 'a', 'toggle'), ['a']);
  assert.deepEqual(selectItems(ids, ['a', 'b'], 'd', 'a', 'single'), ['d']);
  assert.deepEqual(selectItems(ids, ['a'], 'c', 'missing', 'range'), ['c']);
});

test('CSV includes completed errors only, escapes quotes and neutralizes formulas', () => {
  const csv = reportCsv([{ ...item('a'), filename: '=SUM(1,2)".png', status: 'error', error: 'failed' }, item('pending')], 'submission');
  assert.equal(csv.split('\r\n')[0], '﻿path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing');
  assert.equal(csv.split('\r\n')[1], '"\'=SUM(1,2)"".png","","","unknown","","","Failure","0"');
  assert.equal(csv.split('\r\n').length, 3);
});
