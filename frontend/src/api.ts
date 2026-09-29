import type { AnalysisResult, AnatomicalRegion, BatchResult } from './types';

export type ArchiveStreamEvent =
  | { type: 'started'; filename: string; input_position: number; input_path: string }
  | { type: 'result'; item: BatchResult['items'][number] }
  | { type: 'complete'; successful: number; failed: number }
  | { type: 'error'; detail: string };

export async function analyzeFile(
  file: File,
  region: AnatomicalRegion,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  let upload = file;
  if (/\.jpe?g$/i.test(file.name)) {
    // API принимает PNG; JPEG декодируется и преобразуется только локально.
    const bitmap = await createImageBitmap(file);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width; canvas.height = bitmap.height;
      canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
      const png = await new Promise<Blob>((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('Не удалось прочитать JPEG')), 'image/png'));
      upload = new File([png], file.name.replace(/\.jpe?g$/i, '.png'), { type: 'image/png' });
    } finally { bitmap.close(); }
  }
  const form = new FormData();
  form.append('file', upload);
  form.append('anatomical_region', region);

  const response = await fetch('/api/v1/analyses', {
    method: 'POST',
    body: form,
    signal,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? 'Не удалось выполнить анализ');
  }
  const result = (await response.json()) as AnalysisResult;
  return { ...result, filename: file.name };
}

export async function analyzeArchiveIncrementally(
  file: File,
  region: AnatomicalRegion,
  onEvent: (event: ArchiveStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = new FormData();
  form.append('archive', file);
  form.append('anatomical_region', region);
  const response = await fetch('/api/v1/analyses/archive.stream', { method: 'POST', body: form, signal });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? 'Не удалось прочитать архив');
  }
  if (!response.body) throw new Error('Браузер не поддерживает потоковую обработку');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let complete = false;
  function deliver(line: string) {
    const event = JSON.parse(line) as ArchiveStreamEvent;
    if (event.type === 'complete') complete = true;
    onEvent(event);
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      lines.filter(Boolean).forEach(deliver);
      if (done) break;
    }
    if (buffer.trim()) deliver(buffer);
    if (!complete) throw new Error('Поток обработки архива прерван. Добавьте архив повторно.');
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}

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
