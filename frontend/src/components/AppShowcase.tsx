import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import workspace from '../../img/showcase/workspace.webp';
import projects from '../../img/showcase/projects.webp';
import detail from '../../img/showcase/detail.webp';
import Icon from './Icon';
import './AppShowcase.css';

const screenshots = [
  { src: workspace, title: 'Рабочее пространство', alt: 'Интерфейс DEXQ: список снимков, изображение позвоночника с ориентирами и результаты проверок', className: 'showcase-shot--workspace' },
  { src: projects, title: 'Исследования по проектам', alt: 'Список проектов DEXQ с обложками из загруженных снимков', className: 'showcase-shot--projects' },
  { src: detail, title: 'Проверки с объяснением', alt: 'Результаты контроля качества DXA и объяснение выявленного нарушения', className: 'showcase-shot--detail' },
];

export default function AppShowcase() {
  const [active, setActive] = useState<number | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (active !== null) dialogRef.current?.showModal();
  }, [active]);

  return <section className="app-showcase" aria-labelledby="showcase-title">
    <div className="app-showcase__inner">
      <div className="showcase-heading">
        <div><h2 id="showcase-title">От снимка до<br />понятного результата</h2>
          <p>Всё исследование перед глазами. Снимки, ориентиры и проверки — в одном рабочем пространстве.</p></div>
        <Link className="showcase-cta" to="/projects">Открыть DEXQ<Icon name="chevron" size={20} /></Link>
      </div>
      <div className="showcase-composition">
        <span className="showcase-outline" aria-hidden="true" />
        {screenshots.map((shot, index) => <figure key={shot.title} className={`showcase-shot ${shot.className}`}>
          <button type="button" onClick={() => setActive(index)} aria-label={`Увеличить: ${shot.title}`}>
            <img src={shot.src} alt={shot.alt} loading="lazy" decoding="async" />
            <span className="showcase-expand"><Icon name="zoom-in" size={18} /></span>
          </button>
          <figcaption>{shot.title}</figcaption>
        </figure>)}
      </div>
      <div className="showcase-features">
        <article><span className="showcase-feature-icon"><Icon name="folder" size={24} /></span><h3>Исследование целиком</h3><p>Добавляйте снимки перетаскиванием, выбирайте несколько сразу и держите каждое исследование в своём проекте.</p></article>
        <article><span className="showcase-feature-icon"><Icon name="layers" size={24} /></span><h3>Детали под контролем</h3><p>Приближайте изображение, включайте ориентиры и уточняйте ось позвоночника. Любую ручную правку можно отменить.</p></article>
        <article><span className="showcase-feature-icon"><Icon name="download" size={24} /></span><h3>Результат с собой</h3><p>Читайте объяснения проверок и сохраняйте отчёт в CSV или Excel — для всего проекта или выбранных снимков.</p></article>
      </div>
    </div>
    {active !== null && <dialog ref={dialogRef} className="showcase-dialog" onCancel={() => setActive(null)}
      onClick={(event) => { if (event.target === event.currentTarget) setActive(null); }}>
      <div className="showcase-dialog__heading"><strong>{screenshots[active].title}</strong><button autoFocus type="button" aria-label="Закрыть скриншот" onClick={() => setActive(null)}><Icon name="close" size={22} /></button></div>
      <img src={screenshots[active].src} alt={screenshots[active].alt} />
    </dialog>}
  </section>;
}
