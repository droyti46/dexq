import { useEffect, useRef, useState } from 'react';

interface Props {
  onClose: () => void;
  onCreate: (name: string) => void;
}

export default function ProjectDialog({ onClose, onCreate }: Props) {
  const [name, setName] = useState('');
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = dialogRef.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);
  return <dialog ref={dialogRef} className="project-dialog" onCancel={onClose}
    onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <form onSubmit={(event) => { event.preventDefault(); if (name.trim()) onCreate(name); }}>
      <span className="dialog-symbol">DEXQ</span>
      <h2>Новый проект</h2>
      <p>Объедините снимки одного исследования в проект.</p>
      <label htmlFor="project-name">Название проекта</label>
      <input id="project-name" autoFocus required maxLength={80} placeholder="Например, Контроль качества DXA"
        value={name} onChange={(event) => setName(event.target.value)} />
      <small>Проект доступен до закрытия или обновления вкладки.</small>
      <div className="dialog-actions"><button className="quiet-button" type="button" onClick={onClose}>Отмена</button>
        <button className="solid-button" disabled={!name.trim()} type="submit">Создать проект</button></div>
    </form>
  </dialog>;
}
