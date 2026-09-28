export type Point = [number, number];

export function measureManualAxis(
  top: Point, bottom: Point, width: number, height: number,
): { angleDeg: number; violation: boolean } {
  if (![width, height, ...top, ...bottom].every(Number.isFinite) || width <= 0 || height <= 0) {
    throw new Error('Координаты должны быть конечными');
  }
  if ([top, bottom].some(([x, y]) => x < 0 || x > width || y < 0 || y > height)) {
    throw new Error('Точка выходит за пределы снимка');
  }
  const dy = bottom[1] - top[1];
  if (dy <= 0) throw new Error('Верхняя точка должна быть выше нижней');
  const angleDeg = Math.abs(Math.atan2(top[0] - bottom[0], dy) * 180 / Math.PI);
  return { angleDeg, violation: angleDeg > 5 && Math.abs(angleDeg - 5) > 1e-10 };
}

export function pointFromClient(
  clientX: number, clientY: number,
  bounds: { left: number; top: number; width: number; height: number },
  imageWidth: number, imageHeight: number,
): Point {
  if (![clientX, clientY, bounds.left, bounds.top, bounds.width, bounds.height, imageWidth, imageHeight]
    .every(Number.isFinite) || bounds.width <= 0 || bounds.height <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    throw new Error('Размер изображения недоступен');
  }
  const scale = Math.min(bounds.width / imageWidth, bounds.height / imageHeight);
  const visibleWidth = imageWidth * scale;
  const visibleHeight = imageHeight * scale;
  const offsetX = (bounds.width - visibleWidth) / 2;
  const offsetY = (bounds.height - visibleHeight) / 2;
  return [
    Math.max(0, Math.min(imageWidth, (clientX - bounds.left - offsetX) / scale)),
    Math.max(0, Math.min(imageHeight, (clientY - bounds.top - offsetY) / scale)),
  ];
}
