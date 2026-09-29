import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import type { PointerEvent } from 'react';
import { measureManualAxis, pointFromClient } from '../geometry';
import type { Axis } from '../projects';
import type { AnalysisResult, Point } from '../types';
import { zoomAt } from '../viewTransform';
import type { ViewTransform } from '../viewTransform';
import Icon from './Icon';

export interface StudyViewerHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  reset: () => void;
  toggleOverlay: () => void;
}
interface Props {
  result: AnalysisResult;
  manualAxis: Axis | null;
  onAxisChange: (axis: Axis) => void;
  onOverlayChange?: (shown: boolean) => void;
  onAxisCommit?: (before: Axis | null) => void;
}
const initialView: ViewTransform = { zoom: 1, x: 0, y: 0 };

const StudyViewer = forwardRef<StudyViewerHandle, Props>(function StudyViewer({ result, manualAxis, onAxisChange, onOverlayChange, onAxisCommit }, ref) {
  const [overlay, setOverlay] = useState(Boolean(result.geometry || result.annotated_data_url));
  const [tool, setTool] = useState<'pointer' | 'pan'>('pointer');
  const [view, setView] = useState(initialView);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [panning, setPanning] = useState(false);
  const dragRef = useRef<{ pointerId: number; x: number; y: number; start: ViewTransform } | null>(null);
  const axisDragRef = useRef<'top' | 'bottom' | null>(null);
  const axisBeforeRef = useRef<Axis | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const geometry = result.geometry;
  const original = geometry?.axis_line;
  const axis = manualAxis ?? (original ? { top: original.top_xy, bottom: original.bottom_xy } : null);
  const editable = tool === 'pointer' && overlay && result.processing_status === 'Success'
    && result.anatomical_region === 'lumbar_spine' && Boolean(original)
    && result.checks.some((check) => check.check_id === 'spine_axis') && axis !== null;
  const hasOverlay = Boolean(geometry || result.annotated_data_url);

  useEffect(() => {
    setOverlay(Boolean(result.geometry || result.annotated_data_url)); setView(initialView); setBrightness(100); setContrast(100);
    axisDragRef.current = null; dragRef.current = null; setPanning(false);
  }, [result.analysis_id]);
  useEffect(() => { onOverlayChange?.(overlay); }, [overlay, onOverlayChange]);
  useEffect(() => {
    const stage = stageRef.current!;
    const wheel = (event: globalThis.WheelEvent) => {
      event.preventDefault();
      const rect = stage.getBoundingClientRect();
      const cursor = { x: event.clientX - rect.left - rect.width / 2, y: event.clientY - rect.top - rect.height / 2 };
      setView((current) => zoomAt(current, current.zoom * Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.002), cursor));
    };
    stage.addEventListener('wheel', wheel, { passive: false });
    return () => stage.removeEventListener('wheel', wheel);
  }, []);

  function fit() { setView(initialView); }
  function reset() { fit(); setBrightness(100); setContrast(100); setOverlay(hasOverlay); }
  function changeZoom(factor: number) { setView((current) => zoomAt(current, current.zoom * factor, { x: 0, y: 0 })); }
  useImperativeHandle(ref, () => ({
    zoomIn: () => changeZoom(1.25), zoomOut: () => changeZoom(0.8), fit, reset,
    toggleOverlay: () => { if (hasOverlay) setOverlay((current) => !current); },
  }), [hasOverlay]);

  function movePoint(point: Point, which: 'top' | 'bottom') {
    if (!axis || !geometry) return;
    const updated = { ...axis, [which]: point };
    try {
      measureManualAxis(updated.top, updated.bottom, geometry.image_width, geometry.image_height);
      onAxisChange(updated);
    } catch {
      // Пересечение точек или выход за кадр не меняет ручную оценку.
    }
  }
  function handleAxisMove(event: PointerEvent<SVGSVGElement>) {
    if (!axisDragRef.current || !geometry || !svgRef.current) return;
    const point = pointFromClient(event.clientX, event.clientY, svgRef.current.getBoundingClientRect(), geometry.image_width, geometry.image_height);
    movePoint(point, axisDragRef.current);
  }
  function endAxisEdit() {
    if (!axisDragRef.current) return;
    axisDragRef.current = null;
    onAxisCommit?.(axisBeforeRef.current);
  }
  function endPan(event: PointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null; setPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  return <section className="workspace-viewer" aria-label="Просмотр снимка">
    <header className="viewer-header"><div className="viewer-title"><strong title={result.filename}>{result.filename}</strong>
      <span>{result.anatomical_region === 'lumbar_spine' ? 'Поясничный отдел' : result.anatomical_region === 'proximal_femur' ? 'Проксимальный отдел бедра' : 'Область не определена'}</span></div>
      <div className="viewer-tools" role="toolbar" aria-label="Инструменты просмотра">
        <button className={tool === 'pointer' ? 'active' : ''} type="button" onClick={() => setTool('pointer')} title="Редактировать ориентиры" aria-label="Редактировать ориентиры" aria-pressed={tool === 'pointer'}><Icon name="pointer" /></button>
        <button className={tool === 'pan' ? 'active' : ''} type="button" onClick={() => setTool('pan')} title="Перемещение" aria-label="Перемещение" aria-pressed={tool === 'pan'}><Icon name="hand" /></button>
        <span className="viewer-tools__divider" />
        <button type="button" disabled={view.zoom <= 0.25} onClick={() => changeZoom(0.8)} title="Уменьшить" aria-label="Уменьшить"><Icon name="zoom-out" /></button>
        <button className="zoom-value" type="button" onClick={fit} title="Вписать в окно" aria-label={`Масштаб ${Math.round(view.zoom * 100)}%, вписать в окно`}>{Math.round(view.zoom * 100)}%</button>
        <button type="button" disabled={view.zoom >= 8} onClick={() => changeZoom(1.25)} title="Увеличить" aria-label="Увеличить"><Icon name="zoom-in" /></button>
        <button type="button" onClick={fit} title="Вписать в окно" aria-label="Вписать в окно"><Icon name="fit" /></button>
        <span className="viewer-tools__divider" />
        <button className={overlay ? 'active' : ''} disabled={!hasOverlay} type="button" onClick={() => setOverlay(!overlay)} title={overlay ? 'Скрыть наложение' : 'Показать наложение'} aria-label={overlay ? 'Скрыть наложение' : 'Показать наложение'} aria-pressed={overlay}><Icon name="layers" /></button>
        <button type="button" onClick={reset} title="Сбросить вид" aria-label="Сбросить вид"><Icon name="reset" /></button>
      </div>
    </header>
    <div ref={stageRef} className={`viewer-stage ${panning ? 'is-panning' : ''}`} onContextMenu={(event) => event.preventDefault()}
      onPointerDown={(event) => {
        if (![0, 1, 2].includes(event.button)) return;
        event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
        dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, start: view };
        setPanning(true);
      }}
      onPointerMove={(event) => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) return;
        setView({ ...drag.start, x: drag.start.x + event.clientX - drag.x, y: drag.start.y + event.clientY - drag.y });
      }} onPointerUp={endPan} onPointerCancel={endPan} onLostPointerCapture={endPan}
      onDoubleClick={fit}>
      <div className="viewer-canvas" style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)`, transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})` }}>
        <img src={overlay && !geometry ? result.annotated_data_url ?? result.preview_data_url : result.preview_data_url} alt="Обезличенный медицинский снимок" draggable={false} />
        {overlay && geometry && <svg ref={svgRef} viewBox={`0 0 ${geometry.image_width} ${geometry.image_height}`} preserveAspectRatio="xMidYMid meet" aria-label="Ориентиры контроля качества"
          onPointerMove={handleAxisMove} onPointerUp={(event) => {
            endAxisEdit();
            if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          }} onPointerCancel={endAxisEdit} onLostPointerCapture={endAxisEdit}>
          {geometry.gap_lines.map((line, index) => <line key={index} className="viewer-gap" x1={line.endpoints_xy[0][0]} y1={line.endpoints_xy[0][1]} x2={line.endpoints_xy[1][0]} y2={line.endpoints_xy[1][1]} />)}
          {geometry.foreground_bbox && <rect className="viewer-field" x={geometry.foreground_bbox[0]} y={geometry.foreground_bbox[1]} width={geometry.foreground_bbox[2] - geometry.foreground_bbox[0]} height={geometry.foreground_bbox[3] - geometry.foreground_bbox[1]} />}
          {axis && <><line className="viewer-axis" x1={axis.top[0]} y1={axis.top[1]} x2={axis.bottom[0]} y2={axis.bottom[1]} />
            {(['top', 'bottom'] as const).map((which) => <circle key={which} className={editable ? 'viewer-handle' : 'viewer-handle viewer-handle--static'} cx={axis[which][0]} cy={axis[which][1]} r={Math.max(4, geometry.image_width / 80)}
              role={editable ? 'button' : undefined} tabIndex={editable ? 0 : undefined} aria-label={which === 'top' ? 'Верхняя точка оси' : 'Нижняя точка оси'}
              onPointerDown={(event) => {
                if (!editable || event.button !== 0) return;
                event.preventDefault(); event.stopPropagation(); svgRef.current?.setPointerCapture(event.pointerId); axisBeforeRef.current = manualAxis; axisDragRef.current = which;
              }} onKeyDown={(event) => {
                if (!editable || !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                event.preventDefault(); const [x, y] = axis[which];
                movePoint([x + (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0), y + (event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0)], which);
                onAxisCommit?.(manualAxis);
              }} />)}</>}
        </svg>}
      </div>
      <div className="viewer-readout">{geometry && <span>{geometry.image_width} × {geometry.image_height}</span>}<span>{Math.round(view.zoom * 100)}%</span></div>
      <span className="viewer-hint">{editable ? 'Точки — редактирование · ' : ''}Колесо — масштаб · Перетаскивание — перемещение · Двойной клик — вписать</span>
    </div>
    <div className="viewer-adjustments"><label>Яркость <input type="range" min="50" max="150" value={brightness} onChange={(event) => setBrightness(Number(event.target.value))} />{brightness}%</label>
      <label>Контраст <input type="range" min="50" max="150" value={contrast} onChange={(event) => setContrast(Number(event.target.value))} />{contrast}%</label></div>
  </section>;
});
export default StudyViewer;
