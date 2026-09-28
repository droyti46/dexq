import type { BatchResult } from './types';

const key = 'dexq:last-results';
let currentResults: BatchResult | null = null;

type ResultsStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

export function saveResults(results: BatchResult, storage: ResultsStorage = sessionStorage): void {
  currentResults = results;
  try {
    storage.setItem(key, JSON.stringify(results));
  } catch {
    // Объём изображений может превышать квоту браузера; результат остаётся в текущей вкладке.
    try { storage.removeItem(key); } catch { /* Хранилище может быть заблокировано. */ }
  }
}

export function readResults(storage: ResultsStorage = sessionStorage): BatchResult | null {
  if (currentResults) return currentResults;
  try {
    const saved = storage.getItem(key);
    return saved ? JSON.parse(saved) as BatchResult : null;
  } catch {
    return null;
  }
}
