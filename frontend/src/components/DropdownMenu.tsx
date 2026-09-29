import { useEffect, useId, useRef } from 'react';

export interface MenuAction {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  shortcut?: string;
  danger?: boolean;
  divider?: boolean;
}

export default function DropdownMenu({ label, actions, activeMenu, onMenuChange }: {
  label: string; actions: MenuAction[]; activeMenu: string | null; onMenuChange: (label: string | null) => void;
}) {
  const open = activeMenu === label;
  const setOpen = (value: boolean) => onMenuChange(value ? label : null);
  const ref = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const close = (event: globalThis.PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', close);
    const first = ref.current?.querySelector<HTMLButtonElement>('[role="menuitem"]:not(:disabled)');
    (first ?? trigger.current)?.focus();
    return () => document.removeEventListener('pointerdown', close);
  }, [open]);
  return <div className="dropdown" ref={ref} onPointerEnter={() => { if (activeMenu) onMenuChange(label); }} onKeyDown={(event) => {
    if (open && ['ArrowLeft', 'ArrowRight'].includes(event.key)) {
      event.preventDefault();
      const menus = ['Файл', 'Правка', 'Редактор'];
      const index = (menus.indexOf(label) + (event.key === 'ArrowRight' ? 1 : -1) + menus.length) % menus.length;
      onMenuChange(menus[index]);
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus();
    }
    if (!open || !['ArrowDown', 'ArrowUp', 'Home', 'End', 'Tab'].includes(event.key)) return;
    if (event.key === 'Tab') { setOpen(false); return; }
    event.preventDefault();
    const buttons = Array.from(ref.current!.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)'));
    const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
  }}>
    <button ref={trigger} className={`menu-trigger ${open ? 'is-open' : ''}`} type="button" aria-haspopup="menu"
      aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}
      onKeyDown={(event) => { if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); setOpen(true); } }}>{label}</button>
    {open && <div className="dropdown-menu" id={id} role="menu" aria-label={label}>
      {actions.map((action) => <button key={action.label} type="button" role="menuitem" disabled={action.disabled}
        className={`${action.danger ? 'danger-action' : ''} ${action.divider ? 'menu-divider' : ''}`}
        onClick={() => { setOpen(false); trigger.current?.focus(); action.onClick(); }}>
        <span>{action.label}</span>{action.shortcut && <kbd>{action.shortcut}</kbd>}
      </button>)}
    </div>}
  </div>;
}
