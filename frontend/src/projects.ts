import type { AnalysisResult, Point } from './types';
import { reportTable } from './manualAxis.ts';

export type Axis = { top: Point; bottom: Point };
export interface WorkspaceItem {
  id: string;
  file?: File;
  filename: string;
  inputPath?: string;
  size: number;
  position: number;
  status: 'queued' | 'analyzing' | 'ready' | 'error';
  progress: number;
  result?: AnalysisResult;
  savedAxis?: Axis;
  error?: string;
  localPreview?: string;
  archiveId?: string;
}

export interface Project {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  items: WorkspaceItem[];
  manualAxes: Record<string, Axis>;
  axisHistories: Record<string, { past: (Axis | null)[]; present: Axis | null; future: (Axis | null)[] }>;
  paused: boolean;
}

export const maxFiles = 1000;
const extensions = ['.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.zip'];

export function appendFiles(items: WorkspaceItem[], files: File[]): WorkspaceItem[] {
  if (!files.length) return items;
  if (items.length + files.length > maxFiles) throw new Error(`В проекте может быть не более ${maxFiles} снимков.`);
  if (files.some((file) => file.name.toLowerCase().endsWith('.zip')) && files.length !== 1) {
    throw new Error('ZIP-архив нужно добавлять отдельно от остальных снимков.');
  }
  const invalid = files.find((file) => !extensions.some((extension) => file.name.toLowerCase().endsWith(extension)));
  if (invalid) throw new Error(`Формат файла «${invalid.name}» не поддерживается. Используйте DICOM, PNG, JPEG или ZIP.`);
  const position = Math.max(0, ...items.map((item) => item.position));
  return [...items, ...files.map((file, index): WorkspaceItem => ({
    id: crypto.randomUUID(), file, filename: file.name, size: file.size,
    position: position + index + 1, status: 'queued', progress: 0,
  }))];
}

export function updateItem(items: WorkspaceItem[], id: string, changes: Partial<WorkspaceItem>): WorkspaceItem[] {
  return items.map((item) => item.id === id ? { ...item, ...changes } : item);
}

export function removeItems(items: WorkspaceItem[], ids: string[]): WorkspaceItem[] {
  const removed = new Set(ids);
  return items.filter((item) => !removed.has(item.id) && !(item.archiveId && removed.has(item.archiveId)));
}

export function selectItems(
  ids: string[], selected: string[], clicked: string, anchor: string | null, mode: 'single' | 'toggle' | 'range',
): string[] {
  if (mode === 'range' && anchor && ids.includes(anchor)) {
    const a = ids.indexOf(anchor);
    const b = ids.indexOf(clicked);
    return ids.slice(Math.min(a, b), Math.max(a, b) + 1);
  }
  if (mode === 'toggle') return selected.includes(clicked) ? selected.filter((id) => id !== clicked) : [...selected, clicked];
  return [clicked];
}

export function reportCsv(items: WorkspaceItem[], kind: 'submission' | 'clinical' = 'clinical'): string {
  const quote = (value: unknown): string => {
    let text = String(value ?? '');
    if (/^[\s]*[=+\-@\t\r\n]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const { headers, rows } = reportTable(items, kind);
  return '﻿' + headers.join(',') + '\r\n'
    + rows.map((row) => row.map(quote).join(',')).join('\r\n') + '\r\n';
}
