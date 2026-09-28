import { PointerEvent, useEffect, useRef, useState } from 'react';

import { measureManualAxis, pointFromClient } from '../geometry';
import type { AnalysisResult, Point } from '../types';

type Axis = { top: Point; bottom: Point };

interface Props {
  result: AnalysisResult;
  manualAxis: Axis | null;
  onAxisChange: (axis: Axis) => void;
}

export default function StudyViewer({ result, manualAxis, onAxisChange }: Props) {
  const [showAnnotated, setShowAnnotated] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<'top' | 'bottom' | null>(null);
  useEffect(() => { setShowAnnotated(false); setDragging(null); }, [result.analysis_id]);
  const geometry = result.geometry;
  const original = geometry?.axis_line;
  const axis = manualAxis ?? (original ? { top: original.top_xy, bottom: original.bottom_xy } : null);
  const isSpine = result.anatomical_region === 'lumbar_spine';
  const editable = isSpine && axis !== null && !showAnnotated;

  function movePoint(point: Point, which: 'top' | 'bottom') {
    if (!axis || !geometry) return;
    const updated = { ...axis, [which]: point };
    try {
      measureManualAxis(updated.top, updated.bottom, geometry.image_width, geometry.image_height);
      onAxisChange(updated);
    } catch {
      // Некорректный перетаскиваемый промежуточный кадр не меняет ручную оценку.
    }
  }

  function pointerPosition(event: PointerEvent<SVGSVGElement>): Point | null {
    const bounds = svgRef.current?.getBoundingClientRect();
    if (!bounds || !geometry) return null;
    return pointFromClient(event.clientX, event.clientY, bounds, geometry.image_width, geometry.image_height);
  }

  function handleMove(event: PointerEvent<SVGSVGElement>) {
    if (!dragging) return;
    const point = pointerPosition(event);
    if (point) movePoint(point, dragging);
  }

  return (
    <div className="viewer-panel">
      <div className="viewer-panel__toolbar">
        <strong>Снимок и ориентиры</strong>
        {result.annotated_data_url && (
          <button type="button" onClick={() => setShowAnnotated((visible) => !visible)}>
            {showAnnotated ? 'Исходный снимок' : 'Наложение модели'}
          </button>
        )}
      </div>
      <div className="viewer-panel__stage">
        <div className="viewer-panel__canvas">
          {result.preview_data_url ? (
            <img
              src={showAnnotated ? (result.annotated_data_url ?? result.preview_data_url) : result.preview_data_url}
              alt="Изображение DXA без персональных DICOM-тегов"
            />
          ) : <p>Изображение недоступно</p>}
          {!showAnnotated && geometry && result.preview_data_url && (
            <svg
              ref={svgRef}
              viewBox={`0 0 ${geometry.image_width} ${geometry.image_height}`}
              preserveAspectRatio="xMidYMid meet"
              aria-label="Наложение ориентиров в исходных координатах"
              onPointerMove={handleMove}
              onPointerUp={(event) => {
                if (svgRef.current?.hasPointerCapture(event.pointerId)) svgRef.current.releasePointerCapture(event.pointerId);
                setDragging(null);
              }}
              onPointerCancel={() => setDragging(null)}
            >
              {geometry.gap_lines.map((line, index) => (
                <line key={index} className="viewer-gap" x1={line.endpoints_xy[0][0]}
                  y1={line.endpoints_xy[0][1]} x2={line.endpoints_xy[1][0]}
                  y2={line.endpoints_xy[1][1]} />
              ))}
              {geometry.foreground_bbox && (
                <rect className="viewer-field" x={geometry.foreground_bbox[0]}
                  y={geometry.foreground_bbox[1]}
                  width={geometry.foreground_bbox[2] - geometry.foreground_bbox[0]}
                  height={geometry.foreground_bbox[3] - geometry.foreground_bbox[1]} />
              )}
              {axis && <>
                <line className="viewer-axis" x1={axis.top[0]} y1={axis.top[1]}
                  x2={axis.bottom[0]} y2={axis.bottom[1]} />
                {(['top', 'bottom'] as const).map((which) => (
                  <circle key={which} className={editable ? 'viewer-handle' : 'viewer-handle viewer-handle--static'}
                    cx={axis[which][0]} cy={axis[which][1]} r={Math.max(4, geometry.image_width / 80)}
                    role={editable ? 'button' : undefined} tabIndex={editable ? 0 : undefined}
                    aria-label={which === 'top' ? 'Верхняя точка оси' : 'Нижняя точка оси'}
                    onPointerDown={(event) => {
                      if (!editable) return;
                      svgRef.current?.setPointerCapture(event.pointerId);
                      setDragging(which);
                    }}
                    onKeyDown={(event) => {
                      if (!editable || !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                      event.preventDefault();
                      const [x, y] = axis[which];
                      movePoint([
                        x + (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0),
                        y + (event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0),
                      ], which);
                    }}
                  />
                ))}
              </>}
            </svg>
          )}
        </div>
      </div>
      <p className="viewer-panel__caption">
        {isSpine
          ? editable ? 'Перетащите верхнюю или нижнюю точку оси. Исходное решение модели не меняется.'
            : showAnnotated ? 'Контрольное наложение, сформированное эталонным решением.'
              : 'Ось не найдена: ручная правка точек недоступна.'
          : 'Бирюзовая рамка — граница яркостного поля; это не анатомический ROI и не сегментация.'}
      </p>
    </div>
  );
}
