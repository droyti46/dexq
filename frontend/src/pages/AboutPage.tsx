import { Link } from 'react-router-dom';

import AppShell from '../components/AppShell';

export default function AboutPage() {
  return (
    <AppShell>
      <main className="page about-page">
        <section className="about-hero">
          <p className="eyebrow">О продукте</p>
          <h1>Цифровой помощник для контроля DXA</h1>
          <p>
            DEXQ помогает стандартизировать первичную проверку исследования до интерпретации:
            оценивает укладку, охват, артефакты и возвращает объяснимый результат.
          </p>
          <Link className="button button--primary" to="/analyze">
            Открыть демо
          </Link>
        </section>
        <section className="about-principles">
          <article>
            <span>01</span>
            <h2>Модульность</h2>
            <p>Каждый критерий — отдельная проверка. Модели можно менять без изменений интерфейса.</p>
          </article>
          <article>
            <span>02</span>
            <h2>Приватность</h2>
            <p>Инференс выполняется локально. Изображения не передаются во внешние сервисы.</p>
          </article>
          <article>
            <span>03</span>
            <h2>Честный результат</h2>
            <p>Неуверенная или ещё не подключённая проверка обозначается явно, а не маскируется под норму.</p>
          </article>
        </section>
      </main>
    </AppShell>
  );
}

