import writeXlsxFile from 'write-excel-file/browser';
import type { CellObject } from 'write-excel-file/browser';
import type { WorkspaceItem } from './projects';
import { reportTable } from './manualAxis.ts';

export function excelReportData(items: WorkspaceItem[], kind: 'submission' | 'clinical' = 'clinical'): CellObject[][] {
  const { headers, rows } = reportTable(items, kind);
  return [headers.map((value): CellObject => ({ value, type: String, fontWeight: 'bold', textColor: '#FFFFFF', backgroundColor: '#353A42', wrap: true })),
    ...rows.map((row) => row.map((value): CellObject => ({ value, type: typeof value === 'number' ? Number : String, wrap: true, alignVertical: 'top' })))];
}

export async function excelReportBlob(items: WorkspaceItem[], kind: 'submission' | 'clinical' = 'clinical'): Promise<Blob> {
  const data = excelReportData(items, kind);
  return writeXlsxFile(data, {
    sheet: 'DEXQ', stickyRowsCount: 1, columns: [32, 36, 36, 25, 16, 44, 22, 24, 28, 16, 16, 16, 16, 18, 24].slice(0, data[0].length).map((width) => ({ width })),

  }, { fontFamily: 'Arial', fontSize: 11 }).toBlob();
}
