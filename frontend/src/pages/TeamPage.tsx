import alexandrPhoto from '../../img/alexandr_photo.jpg';
import elmirPhoto from '../../img/elmir_photo.jpg';
import nikitaPhoto from '../../img/nikita_photo.jpg';
import AppShell from '../components/AppShell';

const members = [
  {
    name: 'Александр Гладких',
    photo: alexandrPhoto,
    role: 'Машинное обучение · Компьютерное зрение · Backend',
    description: 'Анализ данных, разработка моделей и серверной логики. Отвечает за обработку снимков позвоночника и проверки их качества.',
  },
  {
    name: 'Эльмир Кулиев',
    photo: elmirPhoto,
    role: 'Машинное обучение · Компьютерное зрение',
    description: 'Анализ данных и разработка моделей для исследований таза и проксимального отдела бедра. Отвечает за обработку снимков и проверки качества в этих областях.',
  },
  {
    name: 'Никита Бакутов',
    photo: nikitaPhoto,
    role: 'Full-stack · Дизайн · DevOps',
    description: 'Разработка сайта и пользовательского интерфейса, визуальный дизайн, контейнеризация и развёртывание DEXQ.',
  },
];

export default function TeamPage() {
  return (
    <AppShell>
      <main className="page team-page">
        <section className="page-intro team-intro">
          <p className="eyebrow">Наша команда</p>
          <h1>Команда DEXQ</h1>
          <p>Разрабатываем модели контроля качества DXA и интерфейс для работы с результатами.</p>
        </section>
        <section className="team-grid" aria-label="Участники команды">
          {members.map((member) => (
            <article className="team-card" key={member.name}>
              <img className="team-card__portrait" src={member.photo} alt={member.name} />
              <h2>{member.name}</h2>
              <span className="team-card__role">{member.role}</span>
              <p>{member.description}</p>
            </article>
          ))}
        </section>
      </main>
    </AppShell>
  );
}
