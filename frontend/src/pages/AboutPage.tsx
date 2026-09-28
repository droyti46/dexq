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
            DEXQ локально проверяет качество DXA-снимков до интерпретации специалистом:
            оценивает позиционирование, охват и артефакты. Это исследовательская оценка, не диагноз.
          </p>
          <Link className="button button--primary" to="/analyze">
            Открыть демо
          </Link>
        </section>
        <section className="about-principles">
          <article>
            <span>01</span>
            <h2>Модульность</h2>
            <p>Пять проверок исходного решения сохраняют отдельные ответы; позиционирование и ротация бедра не разделены.</p>
          </article>
          <article>
            <span>02</span>
            <h2>Приватность</h2>
            <p>Инференс выполняется локально. Изображения не передаются во внешние сервисы.</p>
          </article>
          <article>
            <span>03</span>
            <h2>Честный результат</h2>
            <p>Если обязательная проверка не определена — явный отказ, не «норма». Проекцию система не определяет.</p>
          </article>
        </section>
      </main>
    </AppShell>
  );
}

