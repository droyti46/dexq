export interface ViewTransform {
  zoom: number;
  x: number;
  y: number;
}

export function zoomAt(view: ViewTransform, nextZoom: number, cursor: { x: number; y: number }): ViewTransform {
  const zoom = Math.min(8, Math.max(0.25, nextZoom));
  const ratio = zoom / view.zoom;
  return { zoom, x: cursor.x - (cursor.x - view.x) * ratio, y: cursor.y - (cursor.y - view.y) * ratio };
}
