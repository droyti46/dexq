import { useRef } from 'react';
import type { PointerEvent } from 'react';
import { panelWidth } from '../workspaceInteractions';

interface Props {
  side: 'left' | 'right';
  width: number;
  other: number;
  onChange: (width: number) => void;
}

export default function PanelDivider({ side, width, other, onChange }: Props) {
  const dragRef = useRef<{ pointerId: number; x: number; width: number } | null>(null);
  function resize(value: number, element: HTMLElement) {
    const available = element.parentElement!.getBoundingClientRect().width - 44;
    onChange(panelWidth(value, side, available, other));
  }
  function end(event: PointerEvent<HTMLDivElement>) {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  return <div className="panel-divider" role="separator" aria-orientation="vertical" tabIndex={0}
    aria-label={side === 'left' ? 'Ширина панели снимков' : 'Ширина панели результатов'} aria-valuenow={width} aria-valuemin={side === 'left' ? 220 : 260} aria-valuemax={480}
    onPointerDown={(event) => { if (event.button !== 0) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); dragRef.current = { pointerId: event.pointerId, x: event.clientX, width }; }}
    onPointerMove={(event) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      resize(drag.width + (event.clientX - drag.x) * (side === 'left' ? 1 : -1), event.currentTarget);
    }} onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}
    onDoubleClick={() => onChange(side === 'left' ? 270 : 320)}
    onKeyDown={(event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'Home') onChange(side === 'left' ? 270 : 320);
      else resize(width + (event.key === 'ArrowRight' ? 20 : -20) * (side === 'left' ? 1 : -1), event.currentTarget);
    }} />;
}
