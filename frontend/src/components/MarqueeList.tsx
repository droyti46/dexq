import { useEffect, useRef, useState } from 'react';
import type { ReactNode, PointerEvent } from 'react';
import { intersectingItems } from '../workspaceInteractions';

interface Props {
  children: ReactNode;
  selected: string[];
  onSelect: (ids: string[]) => void;
}

export default function MarqueeList({ children, selected, onSelect }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number; clientX: number; clientY: number; base: string[]; moved: boolean } | null>(null);
  const frameRef = useRef(0);
  const suppressClickRef = useRef(false);
  const [box, setBox] = useState<{ left: number; top: number; width: number; height: number } | null>(null);
  useEffect(() => () => {
    dragRef.current = null;
    cancelAnimationFrame(frameRef.current);
  }, []);

  function updateSelection() {
    const list = listRef.current, drag = dragRef.current;
    if (!list || !drag?.moved) return;
    const bounds = list.getBoundingClientRect();
    const end = { x: drag.clientX - bounds.left + list.scrollLeft, y: drag.clientY - bounds.top + list.scrollTop };
    const rows = [...list.querySelectorAll<HTMLElement>('[data-item-id]')].map((row) => {
      const r = row.getBoundingClientRect();
      return { id: row.dataset.itemId!, left: r.left - bounds.left + list.scrollLeft, right: r.right - bounds.left + list.scrollLeft,
        top: r.top - bounds.top + list.scrollTop, bottom: r.bottom - bounds.top + list.scrollTop };
    });
    const ids = intersectingItems(rows, drag, end);
    onSelect([...new Set([...drag.base, ...ids])]);
    setBox({ left: Math.min(drag.x, end.x), top: Math.min(drag.y, end.y), width: Math.abs(end.x - drag.x), height: Math.abs(end.y - drag.y) });
  }
  function autoScroll() {
    const list = listRef.current, drag = dragRef.current;
    if (!list || !drag?.moved) return;
    const bounds = list.getBoundingClientRect();
    const direction = drag.clientY < bounds.top + 28 ? -9 : drag.clientY > bounds.bottom - 28 ? 9 : 0;
    if (direction) { list.scrollTop += direction; updateSelection(); }
    frameRef.current = requestAnimationFrame(autoScroll);
  }
  function end(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    suppressClickRef.current = drag.moved;
    dragRef.current = null; cancelAnimationFrame(frameRef.current); setBox(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    window.setTimeout(() => { suppressClickRef.current = false; }, 0);
  }
  return <div ref={listRef} className="study-list" aria-label="Снимки исследования" title="Перетяните для выделения нескольких снимков"
    onPointerDown={(event) => {
      if (event.button !== 0 || (event.target as HTMLElement).closest('input')) return;
      const rect = event.currentTarget.getBoundingClientRect();
      dragRef.current = { pointerId: event.pointerId, x: event.clientX - rect.left + event.currentTarget.scrollLeft,
        y: event.clientY - rect.top + event.currentTarget.scrollTop, clientX: event.clientX, clientY: event.clientY,
        base: event.ctrlKey || event.metaKey ? selected : [], moved: false };
    }} onPointerMove={(event) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      const distance = Math.hypot(event.clientX - drag.clientX, event.clientY - drag.clientY);
      if (!drag.moved && distance < 5) return;
      if (!drag.moved) { drag.moved = true; event.currentTarget.setPointerCapture(event.pointerId); frameRef.current = requestAnimationFrame(autoScroll); }
      drag.clientX = event.clientX; drag.clientY = event.clientY;
      event.preventDefault(); updateSelection();
    }} onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}
    onClickCapture={(event) => { if (suppressClickRef.current) { event.preventDefault(); event.stopPropagation(); } }}>
    {children}
    {box && <div className="selection-marquee" style={box} />}
  </div>;
}
