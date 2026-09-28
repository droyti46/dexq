import { useMemo, useState } from 'react';
import { Link, Navigate } from 'react-router-dom';

import AppShell from '../components/AppShell';
import StudyViewer from '../components/StudyViewer';
import { measureManualAxis } from '../geometry';
import { readResults } from '../results';
import type { AnalysisResult, CheckResult, Point } from '../types';

const regionLabels = {
  auto: 'Авто',
  lumbar_spine: 'Поясничный отдел',
  proximal_femur: 'Проксимальный отдел бедра',
  unknown: 'Область не определена',
};

const statusLabels: Record<CheckResult['status'], string> = {
  passed: 'Нарушение не обнаружено',
  failed: 'Нарушение',
  not_evaluated: 'Нет оценки',
  error: 'Ошибка',
};

type Axis = { top: Point; bottom: Point };

export default function ResultPage() {
  const batch = useMemo(() => readResults(), []);
  const [activeIndex, setActiveIndex] = useState(0);
  const [page, setPage] = useState(0);
  const [manualAxes, setManualAxes] = useState<Record<string, Axis>>({});
  if (!batch) return <Navigate to="/analyze" replace />;

  const successes = batch.items.filter((item): item is typeof item & { result: AnalysisResult } => item.result?.processing_status === 'Success');
  const errors = batch.items.filter((item) => item.error || item.result?.processing_status === 'Failure');
  const pageSize = 20;
  const result = successes[activeIndex]?.result;
  const manualAxis = result ? manualAxes[result.analysis_id] ?? null : null;
  let manual: { angleDeg: number; violation: boolean } | null = null;
  if (manualAxis && result?.geometry) {
    try {
      manual = measureManualAxis(
        manualAxis.top, manualAxis.bottom, result.geometry.image_width, result.geometry.image_height,
      );
    } catch { /* Невалидные точки не публикуются. */ }
  }
  const manualClass = result && manual
    ? result.checks.some((check) => check.check_id !== 'spine_axis' && check.violation === true)
      || manual.violation ? 1 : result.checks.some((check) => check.check_id !== 'spine_axis' && check.violation === null)
        ? null : 0
    : null;

  function downloadCorrection() {
    if (!result || !manualAxis || !manual) return;
    const cells = [
      result.analysis_id, result.filename, JSON.stringify(manualAxis.top), JSON.stringify(manualAxis.bottom),
      manual.angleDeg.toFixed(4), result.quality_class, manualClass,
    ];
    const quote = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const csv = '﻿analysis_id,filename,top_xy,bottom_xy,manual_angle_deg,model_quality_class,manual_quality_class\n'
      + cells.map(quote).join(',') + '\n';
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    link.download = 'dexq-manual-correction.csv';
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <AppShell>
      <main className="page result-page">
        <div className="result-topline">
          <div>
            <p className="eyebrow">Проверка качества</p>
            <h1>{result?.filename ?? 'Результаты исследования'}</h1>
          </div>
          <Link className="button button--ghost" to="/analyze">Новое исследование</Link>
        </div>

        {errors.length > 0 && (
          <section className="result-errors" aria-label="Файлы без результата">
            <h2>Нужна повторная проверка: {errors.length}</h2>
            <ul>{errors.slice(0, 20).map((item, index) => <li key={`${item.filename}-${index}`}>
              <strong>Изображение {item.input_position ?? index + 1}: {item.filename}</strong>: {item.error || item.result?.error || 'Анализ не завершён'}
            </li>)}</ul>
            {errors.length > 20 && <p>Показаны первые 20 ошибок из {errors.length}. Полный список можно получить через CSV API.</p>}
          </section>
        )}
        {!result && <p className="clinical-note">Нет снимков с завершённой оценкой. Проверьте формат и повторите загрузку.</p>}
        {result && <>
          {successes.length > 1 && <>
            {successes.length > pageSize && <div className="study-pagination">
              <button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>Предыдущие</button>
              <span>Снимки {page * pageSize + 1}–{Math.min((page + 1) * pageSize, successes.length)} из {successes.length}</span>
              <button type="button" disabled={(page + 1) * pageSize >= successes.length} onClick={() => setPage(page + 1)}>Следующие</button>
            </div>}
            <div className="study-tabs" role="tablist" aria-label="Изображения исследования">
              {successes.slice(page * pageSize, (page + 1) * pageSize).map((item, index) => {
                const imageIndex = page * pageSize + index;
                return <button key={`${item.result.analysis_id}-${imageIndex}`} type="button"
                  role="tab" aria-selected={activeIndex === imageIndex} className={activeIndex === imageIndex ? 'active' : ''}
                  onClick={() => setActiveIndex(imageIndex)}>Изображение {item.input_position ?? imageIndex + 1}</button>;
              })}
            </div>
          </>}
          <section className="result-summary">
            <StudyViewer result={result} manualAxis={manualAxis}
              onAxisChange={(axis) => setManualAxes((current) => ({ ...current, [result.analysis_id]: axis }))} />
            <div className="result-overview">
              <div className={`quality-badge quality-badge--${result.quality_class === 1 ? 'failed' : 'passed'}`}>
                <span>{result.quality_class === 1 ? '!' : '✓'}</span>
                <div><small>Автоматическая оценка</small><strong>
                  {result.quality_class === 1 ? 'Есть нарушение качества' : 'Нарушений не выявлено'}
                </strong></div>
              </div>
              {manual && <div className={`manual-verdict ${manualClass === 1 ? 'manual-verdict--failed' : ''}`}>
                <small>После ручного изменения точек</small>
                <strong>{manualClass === 1 ? 'Есть нарушение' : manualClass === 0 ? 'Нарушений не выявлено' : 'Нужна оценка'}</strong>
                <span>Угол {manual.angleDeg.toFixed(2)}° · порог 5°</span>
                <div className="manual-verdict__actions">
                  <button type="button" onClick={() => setManualAxes((current) => {
                    const next = { ...current }; delete next[result.analysis_id]; return next;
                  })}>Сбросить</button>
                  <button type="button" onClick={downloadCorrection}>Скачать ручную оценку</button>
                </div>
              </div>}
              <dl className="metadata-list">
                <div><dt>Область</dt><dd>{regionLabels[result.anatomical_region]}</dd></div>
                <div><dt>Проекция</dt><dd>Не определена для этого набора</dd></div>
                <div><dt>Время</dt><dd>{result.time_of_processing.toFixed(2)} сек</dd></div>
                <div><dt>Study UID</dt><dd>{result.study_uid ?? 'не задан'}</dd></div>
                <div><dt>Image UID</dt><dd>{result.image_uid ?? 'не задан'}</dd></div>
              </dl>
              {result.needs_review && <p className="clinical-note">Маршрутизация или сторона требуют подтверждения специалистом.</p>}
              <p className="clinical-note">Эвристики и модели служат для контроля качества, не для диагностики. Ручной пересчёт не меняет исходный результат модели.</p>
            </div>
          </section>
          <section className="checks-section">
            <div className="section-heading"><div><h2>Проверки качества</h2></div>
              <span>{result.checks.length} проверки</span></div>
            <div className="check-grid">{result.checks.map((check) => <article
              className={`check-card check-card--${check.status}`} key={check.check_id}>
              <div className="check-card__top"><span className="check-status">{statusLabels[check.status]}</span></div>
              <h3>{check.title}</h3><p>{check.summary}</p>
              <footer><span>{check.method === 'reference_qc' ? 'Эталонный алгоритм' : check.method}</span>
                {typeof check.details.score === 'number' && <strong>{check.details.score.toFixed(3)}</strong>}</footer>
            </article>)}</div>
          </section>
        </>}
      </main>
    </AppShell>
  );
}
