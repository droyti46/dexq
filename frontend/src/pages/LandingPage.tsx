import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

import skeleton from '../../img/skeleton.png';
import dots from '../../img/dots.png';
import AppShell from '../components/AppShell';

export default function LandingPage() {
  const heroRef = useRef<HTMLElement>(null);
  const visualRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let frame = 0;
    const update = () => {
      frame = 0;
      if (!heroRef.current || !visualRef.current) return;
      const bounds = heroRef.current.getBoundingClientRect();
      const offset = motion.matches ? 0 : Math.max(0, Math.min(bounds.height, -bounds.top)) * 0.32;
      visualRef.current.style.transform = `translate3d(0, ${offset}px, 0)`;
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    motion.addEventListener('change', schedule);
    update();
    return () => {
      window.removeEventListener('scroll', schedule); window.removeEventListener('resize', schedule);
      motion.removeEventListener('change', schedule); cancelAnimationFrame(frame);
    };
  }, []);
  return (
    <AppShell dark>
      <main ref={heroRef} className="hero">
        <div className="hero__content">
          <p className="eyebrow eyebrow--light">Интеллектуальный контроль качества</p>
          <h1>DEXQ</h1>
          <p className="hero__lead">
            Автоматическая проверка DXA-исследований: укладка, позиционирование,
            артефакты и понятное объяснение нарушений.
          </p>
          <div className="hero__actions">
            <Link className="button button--white" to="/projects">
              Попробовать демо <span aria-hidden="true">↗</span>
            </Link>
            <Link className="text-link" to="/about">
              Как это работает
            </Link>
          </div>
        </div>
        <div ref={visualRef} className="hero__visual" aria-hidden="true">
          <img className="hero__dots" src={dots} alt="" />
          <div className="hero__orb" />
          <img className="hero__skeleton" src={skeleton} alt="" />
        </div>
      </main>
      <section className="promo-section" aria-label="Промо DEXQ">
        <video className="promo-video" controls autoPlay muted playsInline preload="metadata" aria-label="Проморолик DEXQ">
          <source src="/media/dexq-demo.mp4" type="video/mp4" />
          Ваш браузер не поддерживает видео. <a href="/media/dexq-demo.mp4">Открыть ролик</a>
        </video>
      </section>
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
