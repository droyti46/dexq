export function intersectingItems(
  rows: { id: string; left: number; top: number; right: number; bottom: number }[],
  start: { x: number; y: number }, end: { x: number; y: number },
): string[] {
  const left = Math.min(start.x, end.x), right = Math.max(start.x, end.x);
  const top = Math.min(start.y, end.y), bottom = Math.max(start.y, end.y);
  return rows.filter((row) => row.right >= left && row.left <= right && row.bottom >= top && row.top <= bottom).map((row) => row.id);
}

export function panelWidth(requested: number, side: 'left' | 'right', available: number, other: number): number {
  const min = side === 'left' ? 220 : 260;
  return Math.max(min, Math.min(480, available - other - 400, requested));
}

export function estimatedProgress(elapsedMs: number): number {
  return Math.min(96, Math.round(15 + 81 * (1 - Math.exp(-elapsedMs / 2100))));
}
