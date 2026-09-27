import { useMemo, useState } from 'react';
import { Link, Navigate } from 'react-router-dom';

import AppShell from '../components/AppShell';
import type { AnalysisResult, CheckResult } from '../types';

const regionLabels = {
  auto: 'Авто',
  lumbar_spine: 'Поясничный отдел',
  proximal_femur: 'Проксимальный отдел бедра',
  unknown: 'Область не определена',
};

const statusLabels: Record<CheckResult['status'], string> = {
  passed: 'Норма',
  failed: 'Нарушение',
  not_evaluated: 'Нет оценки',
  error: 'Ошибка',
};

export default function ResultPage() {
  const results = useMemo(() => {
    try {
      return JSON.parse(sessionStorage.getItem('dexq:last-results') ?? '[]') as AnalysisResult[];
    } catch {
      return [];
    }
  }, []);
  const [activeIndex, setActiveIndex] = useState(0);
  if (!results.length) return <Navigate to="/analyze" replace />;

  const result = results[activeIndex];
  const isFailed = result.quality_class === 1;
  const isUnknown = result.quality_class === null;
  return (
    <AppShell>
      <main className="page result-page">
        <div className="result-topline">
          <div>
            <p className="eyebrow">Результат анализа</p>
            <h1>{result.filename}</h1>
          </div>
          <Link className="button button--ghost" to="/analyze">
            + Новое исследование
          </Link>
        </div>

        {results.length > 1 && (
          <div className="study-tabs">
            {results.map((item, index) => (
              <button
                type="button"
                className={activeIndex === index ? 'active' : ''}
                onClick={() => setActiveIndex(index)}
                key={item.analysis_id}
              >
                Изображение {index + 1}
              </button>
            ))}
          </div>
        )}

        <section className="result-summary">
          <div className="image-viewer">
            <div className="image-viewer__toolbar">
              <span>{regionLabels[result.anatomical_region]}</span>
              <span>{result.time_of_processing.toFixed(2)} сек</span>
            </div>
            <img src={result.preview_data_url} alt="Обезличенный снимок исследования" />
            <div className="image-viewer__axis" aria-hidden="true" />
          </div>

          <div className="result-overview">
            <div className={`quality-badge quality-badge--${isUnknown ? 'unknown' : isFailed ? 'failed' : 'passed'}`}>
              <span>{isUnknown ? '—' : isFailed ? '!' : '✓'}</span>
              <div>
                <small>Итоговая оценка</small>
                <strong>
                  {isUnknown ? 'Нужна дополнительная оценка' : isFailed ? 'Есть нарушения' : 'Качество соответствует'}
                </strong>
              </div>
            </div>
            <dl className="metadata-list">
              <div>
                <dt>Study UID</dt>
                <dd>{result.study_uid ?? 'нет в файле'}</dd>
              </div>
              <div>
                <dt>Image UID</dt>
                <dd>{result.image_uid ?? result.analysis_id}</dd>
              </div>
              <div>
                <dt>Область</dt>
                <dd>{regionLabels[result.anatomical_region]}</dd>
              </div>
            </dl>
            <p className="clinical-note">
              Результат является инструментом контроля качества и требует подтверждения специалистом.
            </p>
          </div>
        </section>

        <section className="checks-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Детали</p>
              <h2>Проверки качества</h2>
            </div>
            <span>{result.checks.length} модулей</span>
          </div>
          <div className="check-grid">
            {result.checks.map((check, index) => (
              <article className={`check-card check-card--${check.status}`} key={check.check_id}>
                <div className="check-card__top">
                  <span className="check-index">0{index + 1}</span>
                  <span className="check-status">{statusLabels[check.status]}</span>
                </div>
                <h3>{check.title}</h3>
                <p>{check.summary}</p>
                <footer>
                  <span>{check.model_status.replaceAll('_', ' ')}</span>
                  {check.confidence !== null && <strong>{Math.round(check.confidence * 100)}%</strong>}
                </footer>
              </article>
            ))}
          </div>
        </section>
      </main>
    </AppShell>
  );
}

