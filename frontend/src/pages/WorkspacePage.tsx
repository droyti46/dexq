import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, DragEvent, MouseEvent } from 'react';
import { Link, Navigate, useNavigate, useParams } from 'react-router-dom';

import DropdownMenu from '../components/DropdownMenu';
import Icon from '../components/Icon';
import MarqueeList from '../components/MarqueeList';
import PanelDivider from '../components/PanelDivider';
import { excelReportBlob } from '../report';
import { panelWidth } from '../workspaceInteractions';
import ProjectDialog from '../components/ProjectDialog';
import { useProjects } from '../components/ProjectProvider';
import StudyViewer from '../components/StudyViewer';
import type { StudyViewerHandle } from '../components/StudyViewer';
import { measureManualAxis } from '../geometry';
import { reportCsv, selectItems } from '../projects';
import type { Project } from '../projects';
import type { AnatomicalRegion, CheckResult } from '../types';

const regionLabels: Record<AnatomicalRegion, string> = {
  auto: 'Автоопределение', lumbar_spine: 'Поясничный отдел', proximal_femur: 'Проксимальный отдел бедра', unknown: 'Область не определена',
};
const checkStatusLabels: Record<CheckResult['status'], string> = {
  passed: 'Норма', failed: 'Нарушение', not_evaluated: 'Нет оценки', error: 'Ошибка',
};

export default function WorkspacePage() {
  const { projectId } = useParams();
  const { projects } = useProjects();
  const project = projects.find((entry) => entry.id === projectId);
  return project ? <Workspace key={project.id} project={project} /> : <Navigate to="/projects" replace />;
}

function Workspace({ project }: { project: Project }) {
  const { addFiles, deleteItems, togglePause, updateAxis, createProject, commitAxisEdit, restoreAxis } = useProjects();
  const navigate = useNavigate();
  const { items, paused, manualAxes } = project;
  const [activeId, setActiveId] = useState<string | null>(items[0]?.id ?? null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [dragTarget, setDragTarget] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [overlay, setOverlay] = useState(true);
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const [panelSizes, setPanelSizes] = useState({ left: 270, right: 320 });
  const anchorRef = useRef<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const viewerRef = useRef<StudyViewerHandle>(null);
  const deleteDialogRef = useRef<HTMLDialogElement>(null);
  const confirmDeleteRef = useRef<HTMLButtonElement>(null);
  const gridRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const grid = gridRef.current!;
    const clampPanels = () => {
      if (window.innerWidth <= 1100) return;
      const available = grid.getBoundingClientRect().width - 44;
      setPanelSizes((current) => {
        const left = panelWidth(current.left, 'left', available, 260);
        const right = panelWidth(current.right, 'right', available, left);
        return left === current.left && right === current.right ? current : { left, right };
      });
    };
    const observer = new ResizeObserver(clampPanels);
    observer.observe(grid);
    window.addEventListener('resize', clampPanels);
    clampPanels();
    return () => { observer.disconnect(); window.removeEventListener('resize', clampPanels); };
  }, []);

  useEffect(() => {
    document.body.classList.add('workspace-body');
    return () => document.body.classList.remove('workspace-body');
  }, []);
  useEffect(() => {
    setSelectedIds((current) => current.filter((id) => items.some((item) => item.id === id)));
  }, [items]);
  useEffect(() => {
    if (deleting) {
      deleteDialogRef.current?.showModal();
      confirmDeleteRef.current?.focus();
    }
  }, [deleting]);

  const selected = items.find((item) => item.id === activeId) ?? items[0] ?? null;
  const ready = items.filter((item) => item.status === 'ready').length;
  const analyzing = items.filter((item) => item.status === 'analyzing').length;
  const queued = items.filter((item) => item.status === 'queued').length;
  const failed = items.filter((item) => item.status === 'error').length;
  const completed = ready + failed;
  const totalProgress = items.length ? Math.round(items.reduce((sum, item) => sum + item.progress, 0) / items.length) : 0;
  const canView = selected?.status === 'ready' && Boolean(selected.result);
  const canOverlay = canView && Boolean(selected?.result?.geometry || selected?.result?.annotated_data_url);
  const chosen = items.filter((item) => selectedIds.includes(item.id));
  const manualAxis = selected ? manualAxes[selected.id] : undefined;
  const manual = manualAxis && selected?.result?.geometry ? measureManualAxis(manualAxis.top, manualAxis.bottom,
    selected.result.geometry.image_width, selected.result.geometry.image_height) : null;

  function acceptFiles(files: File[]) {
    try { addFiles(project.id, files); setError(''); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : 'Не удалось добавить файлы'); }
  }
  function dropProps(target: string) {
    return {
      onDragEnter: (event: DragEvent<HTMLElement>) => {
        if (event.dataTransfer.types.includes('Files')) { event.preventDefault(); setDragTarget(target); }
      },
      onDragLeave: (event: DragEvent<HTMLElement>) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragTarget(null);
      },
      onDragOver: (event: DragEvent<HTMLElement>) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; },
      onDrop: (event: DragEvent<HTMLElement>) => {
        event.preventDefault(); setDragTarget(null); acceptFiles(Array.from(event.dataTransfer.files));
      },
    };
  }
  function choose(id: string, event: MouseEvent) {
    setActiveId(id);
    setSelectedIds((current) => selectItems(items.map((item) => item.id), current, id, anchorRef.current,
      event.shiftKey ? 'range' : event.ctrlKey || event.metaKey ? 'toggle' : 'single'));
    if (!event.shiftKey) anchorRef.current = id;
  }
  async function download(selectedOnly = false, format: 'csv' | 'xlsx' = 'csv') {
    try {
      const data = selectedOnly ? chosen : items;
      const blob = format === 'xlsx' ? await excelReportBlob(data) : new Blob([reportCsv(data)], { type: 'text/csv;charset=utf-8' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = `dexq-${project.name.replace(/[^\p{L}\p{N}_-]/gu, '_')}${selectedOnly ? '-selected' : ''}.${format}`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { setError('Не удалось экспортировать отчёт. Повторите попытку.'); }
  }
  async function copy(value: string | null) {
    if (!value) return;
    try { await navigator.clipboard.writeText(value); }
    catch { setError('Не удалось скопировать UID. Выделите и скопируйте его вручную.'); }
  }

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (deleting || creating) return;
      if (event.target instanceof Element && event.target.closest('input:not([type="checkbox"]), textarea, select, [contenteditable="true"]')) return;
      if (event.key === 'Delete' && selectedIds.length) {
        event.preventDefault(); setActiveMenu(null); setDeleting(true); return;
      }
      if (!(event.ctrlKey || event.metaKey) || !selected) return;
      const key = event.key.toLowerCase();
      if (key !== 'z' && key !== 'y') return;
      event.preventDefault();
      restoreAxis(project.id, selected.id, key === 'y' || event.shiftKey ? 'redo' : 'undo');
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [project.id, selected?.id, selectedIds.length, restoreAxis, deleting, creating]);
  const history = selected ? project.axisHistories?.[selected.id] : undefined;
  const menuProps = { activeMenu, onMenuChange: setActiveMenu };

  return <div className="workspace-shell">
    <input ref={inputRef} className="workspace-file-input" type="file" multiple tabIndex={-1}
      accept=".dcm,.dicom,.png,.jpg,.jpeg,.zip,application/dicom,application/zip"
      onChange={(event) => { acceptFiles(Array.from(event.target.files ?? [])); event.target.value = ''; }} />
    <header className="app-bar"><div className="app-bar__left">
      <Link className="workspace-wordmark" to="/projects" aria-label="DEXQ — проекты">DEXQ<span /></Link>
      <i className="app-bar__separator" />
      <DropdownMenu {...menuProps} label="Файл" actions={[
        { label: 'Новый проект', onClick: () => setCreating(true) },
        { label: 'Открыть проект…', onClick: () => navigate('/projects') },
        { label: 'Добавить снимки…', onClick: () => inputRef.current?.click(), divider: true },
        { label: 'Экспорт отчёта CSV', onClick: () => void download(), disabled: !completed },
        { label: 'Экспорт отчёта Excel (.xlsx)', onClick: () => void download(false, 'xlsx'), disabled: !completed },
        { label: 'Экспорт выбранных в Excel', onClick: () => void download(true, 'xlsx'), disabled: !chosen.some((item) => item.status === 'ready' || item.status === 'error') },
        { label: 'Выйти в список проектов', onClick: () => navigate('/projects'), divider: true },
      ]} />
      <DropdownMenu {...menuProps} label="Правка" actions={[
        { label: 'Отменить правку оси', onClick: () => selected && restoreAxis(project.id, selected.id, 'undo'), disabled: !history?.past.length, shortcut: 'Ctrl Z' },
        { label: 'Повторить правку оси', onClick: () => selected && restoreAxis(project.id, selected.id, 'redo'), disabled: !history?.future.length, shortcut: 'Ctrl Y' },
        { label: 'Выбрать все', onClick: () => setSelectedIds(items.map((item) => item.id)), disabled: !items.length },
        { label: 'Снять выделение', onClick: () => setSelectedIds([]), disabled: !selectedIds.length },
        { label: 'Удалить выбранные…', onClick: () => setDeleting(true), disabled: !selectedIds.length, danger: true, divider: true, shortcut: 'Del' },
      ]} />
      <DropdownMenu {...menuProps} label="Редактор" actions={[
        { label: 'Увеличить масштаб', onClick: () => viewerRef.current?.zoomIn(), disabled: !canView },
        { label: 'Уменьшить масштаб', onClick: () => viewerRef.current?.zoomOut(), disabled: !canView },
        { label: 'Вписать в окно', onClick: () => viewerRef.current?.fit(), disabled: !canView, divider: true },
        { label: 'Сбросить вид', onClick: () => viewerRef.current?.reset(), disabled: !canView },
        { label: overlay ? 'Скрыть наложение' : 'Показать наложение', onClick: () => viewerRef.current?.toggleOverlay(), disabled: !canOverlay, divider: true },
      ]} />
    </div></header>

    <main ref={gridRef} className="workspace-grid" style={{ '--left-panel': `${panelSizes.left}px`, '--right-panel': `${panelSizes.right}px` } as CSSProperties}>
      <aside className="study-sidebar panel-surface" {...dropProps('sidebar')}>
        <Link className="project-back" to="/projects"><Icon name="back" size={16} />Все проекты</Link>
        <div className="study-card"><div className="study-card__icon"><Icon name="folder" size={24} /></div>
          <div className="study-card__copy"><span>Текущий проект</span><strong title={project.name}>{project.name}</strong>
            <small>{items.length} снимков · {new Date(project.createdAt).toLocaleDateString('ru-RU')}</small></div>
          <button type="button" onClick={() => inputRef.current?.click()} title="Добавить снимки" aria-label="Добавить снимки"><Icon name="plus" /></button>
        </div>
        <div className="selection-bar"><span>{selectedIds.length ? `Выбрано: ${selectedIds.length}` : 'Снимки проекта'}</span>
          {selectedIds.length ? <><button type="button" title="Экспорт выбранных" aria-label="Экспорт выбранных" disabled={!chosen.some((item) => item.status === 'ready' || item.status === 'error')} onClick={() => download(true)}><Icon name="download" size={16} /></button>
            <button type="button" title="Удалить выбранные" aria-label="Удалить выбранные" onClick={() => setDeleting(true)}><Icon name="trash" size={16} /></button>
            <button type="button" title="Снять выделение" aria-label="Снять выделение" onClick={() => setSelectedIds([])}><Icon name="close" size={16} /></button></>
            : <button type="button" title="Выбрать все" onClick={() => setSelectedIds(items.map((item) => item.id))} disabled={!items.length}>Все</button>}
        </div>
        <MarqueeList selected={selectedIds} onSelect={setSelectedIds}>
          {!items.length && <button className="sidebar-empty" type="button" onClick={() => inputRef.current?.click()}><Icon name="upload" size={20} /><strong>Добавить снимки</strong><span>или перетащить сюда · до 200</span></button>}
          {items.map((item) => {
            const quality = item.result?.quality_class;
            return <div key={item.id} data-item-id={item.id} className={`study-item ${selected?.id === item.id ? 'active' : ''} ${selectedIds.includes(item.id) ? 'is-selected' : ''}`}>
              <input className="study-checkbox" type="checkbox" aria-label={`Выбрать ${item.filename}`} checked={selectedIds.includes(item.id)}
                onChange={() => { setSelectedIds((current) => selectItems(items.map((entry) => entry.id), current, item.id, null, 'toggle')); anchorRef.current = item.id; }} />
              <button className="study-item__open" type="button" aria-label={`Открыть ${item.filename}`} aria-current={selected?.id === item.id ? 'true' : undefined} onClick={(event) => choose(item.id, event)}>
                <div className="study-item__thumb">{(item.result?.preview_data_url || item.localPreview) ? <img src={item.result?.preview_data_url || item.localPreview} alt="" draggable={false} /> : <span>DX</span>}</div>
                <div className="study-item__content"><strong title={item.filename}>{item.filename}</strong><span>Изображение {item.position}</span>
                  <div className={`study-item__status status-${item.status}`}>
                    <span className={`status-icon status-icon--${item.status} ${quality === 1 ? 'status-icon--violation' : ''}`}>
                      {item.status === 'ready' ? <Icon name={quality === 1 ? 'alert' : 'check'} size={14} /> : item.status === 'analyzing' ? <i /> : <Icon name={item.status === 'error' ? 'alert' : 'clock'} size={14} />}
                    </span>
                    <em>{item.status === 'ready' ? quality === 1 ? 'Есть нарушение' : 'Норма' : item.status === 'analyzing' ? 'Анализируется' : item.status === 'error' ? 'Ошибка' : 'В очереди'}</em>
                    {item.status === 'analyzing' && <small>{item.progress}%</small>}
                  </div></div>
              </button>
            </div>;
          })}
        </MarqueeList>
        <button className="sidebar-add" type="button" onClick={() => inputRef.current?.click()}><Icon name="plus" size={16} />Добавить снимки</button>
        {dragTarget === 'sidebar' && <div className="workspace-drop-overlay"><Icon name="upload" size={28} /><strong>Добавить в проект</strong><span>Текущие снимки останутся</span></div>}
      </aside>

      <PanelDivider side="left" width={panelSizes.left} other={panelSizes.right} onChange={(left) => setPanelSizes((current) => ({ ...current, left }))} />
      <section className="viewer-column panel-surface" {...dropProps('viewer')}>
        {dragTarget === 'viewer' && <div className="workspace-drop-overlay"><Icon name="upload" size={30} /><strong>Добавить в проект</strong><span>Текущие снимки останутся</span></div>}
        {selected?.status === 'ready' && selected.result ? <StudyViewer ref={viewerRef} key={selected.id} result={selected.result} manualAxis={manualAxis ?? null}
          onOverlayChange={setOverlay} onAxisChange={(axis) => updateAxis(project.id, selected.id, axis)} onAxisCommit={(before) => commitAxisEdit(project.id, selected.id, before)} /> : selected ? (
          <div className={`viewer-placeholder viewer-placeholder--${selected.status}`}>
            {selected.localPreview && <img src={selected.localPreview} alt="Предпросмотр загруженного изображения" />}
            <div className="viewer-placeholder__message">{selected.status === 'analyzing' ? <><span className="analysis-spinner" /><strong>Анализируем изображение</strong><p>Результат появится здесь после обработки</p></>
              : selected.status === 'error' ? <><Icon name="alert" size={34} /><strong>Не удалось обработать снимок</strong><p>{selected.error}</p></>
                : <><Icon name="clock" size={34} /><strong>{paused ? 'Обработка на паузе' : 'Снимок в очереди'}</strong><p>Можно просматривать готовые снимки или добавлять новые.</p></>}</div>
          </div>) : <button className="workspace-empty" type="button" onClick={() => inputRef.current?.click()}><span><Icon name="upload" size={28} /></span><strong>Добавьте снимки в проект</strong><p>DICOM, PNG, JPEG · ZIP с DICOM или PNG</p><em>Выбрать файлы</em></button>}
      </section>

      <PanelDivider side="right" width={panelSizes.right} other={panelSizes.left} onChange={(right) => setPanelSizes((current) => ({ ...current, right }))} />
      <aside className="inspector panel-surface">
        {selected?.status === 'ready' && selected.result ? <>
          <div className={`quality-summary quality-summary--${selected.result.quality_class === 1 ? 'failed' : 'passed'}`}><span><Icon name={selected.result.quality_class === 1 ? 'alert' : 'check'} size={24} /></span>
            <div><small>Автоматическая оценка</small><strong>{selected.result.quality_class === 1 ? 'Есть нарушение качества' : 'Нарушений не выявлено'}</strong><p>{selected.result.checks.filter((check) => check.violation === true).length} нарушений · {selected.result.checks.length} проверок</p></div></div>
          <section className="inspector-section"><header><h2>Параметры снимка</h2></header><dl className="inspector-metadata">
            <div><dt>Область</dt><dd>{regionLabels[selected.result.anatomical_region]}</dd></div>
            <div><dt>Проекция</dt><dd>Не определена</dd></div>
            <div><dt>Время анализа</dt><dd>{selected.result.time_of_processing.toFixed(2)} сек</dd></div>
            <div><dt>Study UID</dt><dd title={selected.result.study_uid ?? ''}>{selected.result.study_uid ?? 'Не задан'}</dd><button type="button" aria-label="Скопировать Study UID" disabled={!selected.result.study_uid} onClick={() => void copy(selected.result!.study_uid)}><Icon name="copy" size={15} /></button></div>
            <div><dt>Image UID</dt><dd title={selected.result.image_uid ?? ''}>{selected.result.image_uid ?? 'Не задан'}</dd><button type="button" aria-label="Скопировать Image UID" disabled={!selected.result.image_uid} onClick={() => void copy(selected.result!.image_uid)}><Icon name="copy" size={15} /></button></div>
          </dl></section>
          <section className="inspector-section inspector-checks"><header><h2>Проверки качества</h2></header><div>
            {selected.result.checks.map((check) => <details key={check.check_id} className={`check-detail check-row--${check.status}`}><summary className="check-row">
              <span className="check-row__icon"><Icon name={check.status === 'passed' ? 'check' : 'alert'} size={16} /></span><strong>{check.title}</strong><em>{checkStatusLabels[check.status]}</em><Icon name="chevron" size={14} />
            </summary><p>{check.summary}</p></details>)}
          </div></section>
          {manual && <div className="manual-note"><strong>Ручная ось: {manual.angleDeg.toFixed(2)}°</strong><p>Эвристика: отклонение больше 5°. Автоматическая оценка не изменена.</p><button className="quiet-button" type="button" onClick={() => { updateAxis(project.id, selected.id, null); commitAxisEdit(project.id, selected.id, manualAxis ?? null); }}>Сбросить ручную ось</button></div>}
          {selected.result.needs_review && <p className="inspector-note">Нужна проверка специалистом.</p>}
          <p className="inspector-note">Исследовательский контроль качества, не диагностика. Автоматический результат требует экспертной проверки.</p>
        </> : <div className="inspector-empty"><Icon name="layers" size={24} /><strong>{selected ? 'Ожидаем результат' : 'Параметры снимка'}</strong><p>{selected ? 'Проверки появятся после анализа.' : 'Выберите снимок, чтобы посмотреть оценку и параметры.'}</p></div>}
      </aside>
    </main>

    <footer className="processing-bar"><div className="processing-overview"><div><strong>{items.length ? paused ? 'Обработка на паузе' : analyzing || queued ? 'Обработка снимков' : 'Обработка завершена' : 'Готовы к работе'}</strong><span>{completed} из {items.length}</span></div>
      <div className="linear-progress" title="Приблизительный прогресс; 100% после завершения анализа" role="progressbar" aria-label="Обработка проекта" aria-valuenow={totalProgress} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${totalProgress}%` }} /></div><b>{totalProgress}%</b></div>
      <div className="processing-stat processing-stat--ready"><Icon name="check" size={16} /><strong>{ready}</strong><span>готово</span></div>
      <div className="processing-stat"><Icon name="clock" size={16} /><strong>{analyzing + queued}</strong><span>в обработке</span></div>
      <div className="processing-stat processing-stat--error"><Icon name="alert" size={16} /><strong>{failed}</strong><span>ошибок</span></div>
      {(analyzing > 0 || queued > 0) && <button className="pause-button" type="button" onClick={() => togglePause(project.id)} title="Пауза после текущего снимка"><Icon name={paused ? 'play' : 'pause'} size={16} />{paused ? 'Продолжить' : 'Пауза'}</button>}
    </footer>
    {error && <div className="workspace-toast" role="alert"><Icon name="alert" /><span>{error}</span><button type="button" aria-label="Закрыть сообщение" onClick={() => setError('')}><Icon name="close" /></button></div>}
    {creating && <ProjectDialog onClose={() => setCreating(false)} onCreate={(name) => navigate(`/projects/${createProject(name)}`)} />}
    {deleting && <dialog ref={deleteDialogRef} className="project-dialog" onCancel={() => setDeleting(false)} onClick={(event) => { if (event.target === event.currentTarget) setDeleting(false); }}>
      <div><h2>Удалить выбранные снимки?</h2><p>Выбрано: {selectedIds.length}. Снимки и их результаты будут удалены из проекта. Исходные файлы на устройстве останутся.</p>
        <div className="dialog-actions"><button className="quiet-button" type="button" onClick={() => setDeleting(false)}>Отмена</button><button ref={confirmDeleteRef} className="danger-button" autoFocus type="button" onClick={() => { deleteItems(project.id, selectedIds); setSelectedIds([]); setDeleting(false); }}>Удалить снимки</button></div></div>
    </dialog>}
  </div>;
}
