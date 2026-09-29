import writeXlsxFile from 'write-excel-file/browser';
import type { CellObject } from 'write-excel-file/browser';
import type { WorkspaceItem } from './projects';

export function excelReportData(items: WorkspaceItem[]): CellObject[][] {
  const headers = ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region', 'quality_class', 'violation_type', 'processing_status', 'time_of_processing'];
  const rows = items.filter((item) => item.status === 'ready' || item.status === 'error').map((item) => {
    const r = item.result;
    return [item.filename, r?.study_uid ?? '', r?.image_uid ?? '', r?.anatomical_region ?? 'unknown',
      r?.quality_class ?? '', r?.violation_types.join(';') ?? '', r?.processing_status ?? 'Failure', r?.time_of_processing ?? '']
      .map((value): CellObject => ({ value, type: typeof value === 'number' ? Number : String, wrap: true, alignVertical: 'top' }));
  });
  return [headers.map((value): CellObject => ({ value, type: String, fontWeight: 'bold', textColor: '#FFFFFF', backgroundColor: '#353A42', wrap: true })), ...rows];
}

export async function excelReportBlob(items: WorkspaceItem[]): Promise<Blob> {
  return writeXlsxFile(excelReportData(items), {
    sheet: 'DEXQ', stickyRowsCount: 1, columns: [32, 36, 36, 25, 16, 44, 22, 24].map((width) => ({ width })),
  }, { fontFamily: 'Arial', fontSize: 11 }).toBlob();
}
