import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Icon from '../components/Icon';
import ProjectDialog from '../components/ProjectDialog';
import { useProjects } from '../components/ProjectProvider';

export default function ProjectsPage() {
  const { projects, createProject } = useProjects();
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();
  return <div className="projects-shell">
    <header className="projects-header"><Link className="workspace-wordmark" to="/" aria-label="DEXQ — главная">DEXQ<span /></Link>
      <span>Контроль качества DXA</span><Link className="quiet-button" to="/">На главную</Link></header>
    <main className="projects-main">
      <div className="projects-heading"><div><h1>Ваши проекты</h1><p>Снимки, результаты и рабочее пространство одного исследования.</p></div>
        <button className="solid-button" type="button" onClick={() => setCreating(true)}><Icon name="plus" />Создать проект</button></div>
      <div className="session-note"><Icon name="lock" size={16} /><span>Только в этой вкладке. Перед обновлением или закрытием сохраните отчёт.</span></div>
      <section className="project-grid" aria-label="Проекты">
        <button className="project-new-card" type="button" onClick={() => setCreating(true)}>
          <span><Icon name="plus" size={26} /></span><strong>Новый проект</strong><small>Добавьте снимки и начните анализ</small>
        </button>
        {projects.map((project) => {
          const previews = project.items.map((item) => item.result?.preview_data_url || item.localPreview).filter((url): url is string => Boolean(url)).slice(0, 3);
          const pending = project.items.filter((item) => item.status === 'queued' || item.status === 'analyzing').length;
          return <Link className="project-card" key={project.id} to={`/projects/${project.id}`} aria-label={`Открыть проект ${project.name}`}>
            <div className={`project-cover ${previews.length ? 'has-images' : ''}`}>
              {previews.length ? <div className="preview-stack">{previews.map((url, index) => <img src={url} key={index} alt="" />)}</div>
                : <Icon name="folder" size={44} />}
              <span className="project-cover__count">{project.items.length} снимков</span>
              {pending > 0 && <span className="project-cover__status">{project.paused ? 'Пауза' : 'Обработка'}</span>}
            </div>
            <div className="project-card__info"><strong>{project.name}</strong><span>{new Date(project.updatedAt).toLocaleString('ru-RU', { day: 'numeric', month: 'long', hour: '2-digit', minute: '2-digit' })}<Icon name="chevron" size={16} /></span></div>
          </Link>;
        })}
      </section>
      {!projects.length && <p className="projects-empty-note">Пока нет проектов. Создайте первый и добавьте DICOM, PNG или ZIP-архив.</p>}
    </main>
    <footer className="projects-footer">Локальная обработка · Исследовательский инструмент, не для диагностики</footer>
    {creating && <ProjectDialog onClose={() => setCreating(false)} onCreate={(name) => navigate(`/projects/${createProject(name)}`)} />}
  </div>;
}
