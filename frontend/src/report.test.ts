import assert from 'node:assert/strict';
import test from 'node:test';
import { excelReportData, excelReportBlob } from './report.ts';
import type { WorkspaceItem } from './projects.ts';
const completed: WorkspaceItem = { id: 'a', filename: '=SUM(1,2).png', size: 0, position: 1, status: 'error', progress: 100 };

test('Excel exports same eight columns with literal filenames and no pending rows', () => {
  const data = excelReportData([completed, { ...completed, id: 'b', status: 'queued' }], 'submission');
  assert.equal(data.length, 2);
  assert.equal(data[0].length, 8);
  assert.equal(data[1][0].value, '=SUM(1,2).png');
  assert.equal(data[1][0].type, String);
  assert.equal(data[1][6].value, 'Failure');
});
test('Excel output is a genuine zipped XLSX rather than renamed CSV', async () => {
  const blob = await excelReportBlob([completed]);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  assert.equal(String.fromCharCode(...bytes.slice(0, 2)), 'PK');
  assert.ok(bytes.length > 1000);
});
