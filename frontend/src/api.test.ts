import assert from 'node:assert/strict';
import test from 'node:test';
import { analyzeArchiveIncrementally } from './api.ts';

async function withResponse(body: string, action: () => Promise<void>) {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(body, { status: 200 });
  try { await action(); } finally { globalThis.fetch = original; }
}

test('truncated archive stream is an error rather than success', async () => {
  await withResponse('{"type":"started","filename":"a.png","input_position":1}\n', async () => {
    await assert.rejects(analyzeArchiveIncrementally(new File(['zip'], 'synthetic.zip'), 'auto', () => {}), /прерван/);
  });
});
test('complete archive stream delivers final event even without newline', async () => {
  await withResponse('{"type":"complete","successful":0,"failed":0}', async () => {
    const events: string[] = [];
    await analyzeArchiveIncrementally(new File(['zip'], 'synthetic.zip'), 'auto', (event) => events.push(event.type));
    assert.deepEqual(events, ['complete']);
  });
});
