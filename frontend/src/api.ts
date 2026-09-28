import type { AnalysisResult, AnatomicalRegion, BatchResult } from './types';

export async function analyzeStudies(
  files: File[],
  region: AnatomicalRegion,
): Promise<BatchResult> {
  const form = new FormData();
  const isArchive = files.length === 1 && files[0].name.toLowerCase().endsWith('.zip');
  const isBatch = files.length > 1;
  files.forEach((file) => form.append(isArchive ? 'archive' : isBatch ? 'files' : 'file', file));
  form.append('anatomical_region', region);

  const response = await fetch(isArchive ? '/api/v1/analyses/archive' : isBatch ? '/api/v1/analyses/batch' : '/api/v1/analyses', {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? 'Не удалось выполнить анализ');
  }
  if (!isArchive && !isBatch) {
    const result = (await response.json()) as AnalysisResult;
    return { items: [{ filename: result.filename, input_position: 1, result, error: null }], successful: 1, failed: 0 };
  }
  return (await response.json()) as BatchResult;
}

