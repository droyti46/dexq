import { Link } from 'react-router-dom';

import skeleton from '../../img/skeleton.png';
import dots from '../../img/dots.png';
import AppShell from '../components/AppShell';

export default function LandingPage() {
  return (
    <AppShell dark>
      <main className="hero">
        <div className="hero__content">
          <p className="eyebrow eyebrow--light">Интеллектуальный контроль качества</p>
          <h1>DEXQ</h1>
          <p className="hero__lead">
            Автоматическая проверка DXA-исследований: укладка, позиционирование,
            артефакты и понятное объяснение нарушений.
          </p>
          <div className="hero__actions">
            <Link className="button button--white" to="/analyze">
              Попробовать демо <span aria-hidden="true">↗</span>
            </Link>
            <Link className="text-link" to="/about">
              Как это работает
            </Link>
          </div>
        </div>
        <div className="hero__visual" aria-hidden="true">
          <img className="hero__dots" src={dots} alt="" />
          <div className="hero__orb" />
          <img className="hero__skeleton" src={skeleton} alt="" />
        </div>
      </main>
      <section className="feature-strip" aria-label="Основные возможности">
        <article>
          <span>01</span>
          <h2>До 3 серий</h2>
          <p>Один файл или пакет исследования.</p>
        </article>
        <article>
          <span>02</span>
          <h2>6 проверок</h2>
          <p>Независимые модули без связи с интерфейсом.</p>
        </article>
        <article>
          <span>03</span>
          <h2>Понятный отчёт</h2>
          <p>Тип нарушения, уверенность и объяснение.</p>
        </article>
      </section>
    </AppShell>
  );
}
