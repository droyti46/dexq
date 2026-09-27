import AppShell from '../components/AppShell';

const roles = [
  {
    initials: 'CV',
    role: 'Computer Vision',
    description: 'Модели локализации анатомических ориентиров и классификации нарушений.',
  },
  {
    initials: 'ML',
    role: 'Machine Learning',
    description: 'Подготовка данных, воспроизводимые эксперименты и оценка качества.',
  },
  {
    initials: 'DEV',
    role: 'Product & Engineering',
    description: 'Локальный сервис, медицинский интерфейс и контейнеризация решения.',
  },
];

export default function TeamPage() {
  return (
    <AppShell>
      <main className="page team-page">
        <section className="page-intro team-intro">
          <p className="eyebrow">Наша команда</p>
          <h1>Объединяем медицину, данные и инженерный подход</h1>
          <p>
            Пока здесь нейтральный шаблон. Имена, фотографии и роли можно заменить перед
            презентацией без изменения структуры страницы.
          </p>
        </section>
        <section className="team-grid">
          {roles.map((member, index) => (
            <article className="team-card" key={member.role}>
              <div className="team-card__portrait">{member.initials}</div>
              <span>0{index + 1}</span>
              <h2>{member.role}</h2>
              <p>{member.description}</p>
            </article>
          ))}
        </section>
      </main>
    </AppShell>
  );
}
