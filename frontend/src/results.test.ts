import assert from 'node:assert/strict';
import test from 'node:test';

import { readResults, saveResults } from './results.ts';
import type { BatchResult } from './types.ts';

test('clears stale stored results when a new response exceeds quota', () => {
  let stored: string | null = 'previous result';
  const storage = {
    getItem: () => stored,
    setItem: () => { throw new DOMException('Quota exceeded', 'QuotaExceededError'); },
    removeItem: () => { stored = null; },
  };
  saveResults({ items: [], successful: 0, failed: 0 }, storage);
  assert.equal(stored, null);
});

test('keeps analyzed response available when session quota is exhausted', () => {
  const batch: BatchResult = {
    items: [{ filename: 'study.png', input_position: 1, result: null, error: 'Ошибка' }],
    successful: 0,
    failed: 1,
  };
  const storage = {
    getItem: () => null,
    setItem: () => { throw new DOMException('Quota exceeded', 'QuotaExceededError'); },
    removeItem: () => {},
  };
  saveResults(batch, storage);
  assert.deepEqual(readResults(storage), batch);
});
