import type { AnalysisResult, AnatomicalRegion, BatchResult } from './types';

export async function analyzeStudies(
  files: File[],
  region: AnatomicalRegion,
): Promise<BatchResult> {
  const form = new FormData();
  const isBatch = files.length > 1;
  files.forEach((file) => form.append(isBatch ? 'files' : 'file', file));
  form.append('anatomical_region', region);

  const response = await fetch(isBatch ? '/api/v1/analyses/batch' : '/api/v1/analyses', {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? 'Не удалось выполнить анализ');
  }
  if (!isBatch) {
    const result = (await response.json()) as AnalysisResult;
    return { items: [{ filename: result.filename, result, error: null }], successful: 1, failed: 0 };
  }
  return (await response.json()) as BatchResult;
}

