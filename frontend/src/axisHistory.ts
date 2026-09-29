import type { Axis } from './projects';

export interface AxisHistory {
  past: (Axis | null)[];
  present: Axis | null;
  future: (Axis | null)[];
}

export function commitAxis(history: AxisHistory, before: Axis | null): AxisHistory {
  if (JSON.stringify(before) === JSON.stringify(history.present)) return history;
  return { past: [...history.past, before].slice(-100), present: history.present, future: [] };
}
export function undoAxis(history: AxisHistory): AxisHistory {
  if (!history.past.length) return history;
  return { past: history.past.slice(0, -1), present: history.past[history.past.length - 1], future: [history.present, ...history.future] };
}
export function redoAxis(history: AxisHistory): AxisHistory {
  if (!history.future.length) return history;
  return { past: [...history.past, history.present], present: history.future[0], future: history.future.slice(1) };
}
