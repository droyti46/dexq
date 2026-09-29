import { Link } from 'react-router-dom';

import AppShell from '../components/AppShell';
import './AboutPage.css';

export default function AboutPage() {
  return (
    <AppShell>
      <main className="page about-page">
        <section className="about-intro" aria-labelledby="about-title">
          <div className="about-intro__content">
            <h1 id="about-title">Контроль качества DXA до заключения врача</h1>
            <p className="about-intro__lead">
              DEXQ проверяет снимки позвоночника и бедра, показывает обнаруженные нарушения
              и помогает подготовить отчёт. Анализ выполняется локально.
            </p>
            <Link className="button button--white" to="/projects">Попробовать демо</Link>
          </div>
          <p className="about-intro__note">
            Исследовательский инструмент. Оценка не является диагнозом и не прошла
            независимую клиническую валидацию.
          </p>
        </section>

        <section className="about-section" aria-labelledby="about-process-title">
          <div className="about-section__heading">
            <h2 id="about-process-title">От снимка до отчёта</h2>
          </div>
          <ol className="about-steps">
            <li>
              <span className="about-steps__number" aria-hidden="true">01</span>
              <div>
                <h3>Загрузите исследование</h3>
                <p>Создайте проект и добавьте обезличенные DICOM, PNG или JPEG либо ZIP с DICOM/PNG. Обработка идёт локально: снимки не уходят во внешние сервисы.</p>
              </div>
            </li>
            <li>
              <span className="about-steps__number" aria-hidden="true">02</span>
              <div>
                <h3>Проверьте результат</h3>
                <p>Для каждого снимка показаны область, оценка качества и результаты проверок. Если оценка невозможна, вместо отметки «норма» появится ошибка.</p>
              </div>
            </li>
            <li>
              <span className="about-steps__number" aria-hidden="true">03</span>
              <div>
                <h3>Уточните ось позвоночника</h3>
                <p>Передвиньте найденные точки: угол и предварительная оценка обновятся сразу. Нажмите «Сохранить», чтобы правка вошла в отчёт врача. Порог 5° — эвристика; автоматический ответ остаётся неизменным.</p>
              </div>
            </li>
            <li>
              <span className="about-steps__number" aria-hidden="true">04</span>
              <div>
                <h3>Скачайте нужный отчёт</h3>
                <p>«Для организаторов» — автоматический результат в CSV или Excel. «Отчёт врача» включает сохранённые правки оси. Ошибки обработки тоже отражаются в отчёте.</p>
              </div>
            </li>
          </ol>
        </section>

        <section className="about-section about-checks" aria-labelledby="about-checks-title">
          <div className="about-section__heading">
            <h2 id="about-checks-title">Две области, разные признаки</h2>
          </div>
          <div className="about-checks__grid">
            <article>
              <p className="about-checks__area">Позвоночник</p>
              <h3>Охват, ось, артефакты</h3>
              <p>Охват оценивается эвристически, ось — по двум найденным точкам; артефакты проверяются отдельно. Редактировать ось можно, если точки обнаружены.</p>
            </article>
            <article>
              <p className="about-checks__area">Бедро</p>
              <h3>Укладка и охват поля</h3>
              <p>Укладка и ротация оцениваются вместе. Рамка показывает границы яркостного поля снимка, а не анатомическую маску.</p>
            </article>
          </div>
        </section>

        <aside className="about-boundaries" aria-label="Пределы работы и хранение">
          <div>
            <h2>До 1000 снимков в проекте</h2>
            <p>Добавьте снимки по отдельности или одним ZIP-архивом.</p>
          </div>
          <div>
            <h2>Сохраните отчёт</h2>
            <p>Проекты и правки доступны до обновления или закрытия вкладки.</p>
          </div>
        </aside>
      </main>
    </AppShell>
  );
}
