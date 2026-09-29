import { measureManualAxis } from './geometry.ts';
import type { Axis, WorkspaceItem } from './projects';
import type { AnalysisResult } from './types';

function resultWithAxis(item: WorkspaceItem, axis: Axis | null): AnalysisResult | undefined {
  const { result } = item;
  if (!axis || item.status !== 'ready' || result?.processing_status !== 'Success'
    || result.anatomical_region !== 'lumbar_spine' || !result.geometry?.axis_line
    || !result.checks.some((check) => check.check_id === 'spine_axis')) return result;
  const manual = measureManualAxis(axis.top, axis.bottom, result.geometry.image_width, result.geometry.image_height);
  const checks = result.checks.map((check) => check.check_id === 'spine_axis' ? {
    ...check, status: manual.violation ? 'failed' as const : 'passed' as const, violation: manual.violation,
    confidence: null, method: 'manual_axis_heuristic', model_status: 'not_used',
    summary: `Ручная ось: ${manual.angleDeg.toFixed(2)}°. Эвристика: нарушение при отклонении больше 5°.`,
    details: { angle_deg: manual.angleDeg, threshold_deg: 5, source: 'manual' },
  } : check);
  const violations = new Set(result.violation_types.filter((id) => id !== 'spine_axis'));
  if (manual.violation) violations.add('spine_axis');
  return { ...result, checks, violation_types: [...violations].sort(), quality_class: violations.size ? 1 : 0,
    needs_review: true,
    geometry: { ...result.geometry, axis_line: { top_xy: axis.top, bottom_xy: axis.bottom, angle_deg: manual.angleDeg } },
  };
}

export function resultWithSavedAxis(item: WorkspaceItem): AnalysisResult | undefined {
  return resultWithAxis(item, item.savedAxis ?? null);
}

export function resultWithDraftAxis(item: WorkspaceItem, draft: Axis | null): AnalysisResult | undefined {
  return resultWithAxis(item, draft);
}

export function axesEqual(a: Axis | null, b: Axis | null): boolean {
  return a === b || (a !== null && b !== null
    && a.top[0] === b.top[0] && a.top[1] === b.top[1]
    && a.bottom[0] === b.bottom[0] && a.bottom[1] === b.bottom[1]);
}

export type ReportKind = 'submission' | 'clinical';

export function validateSubmission(items: WorkspaceItem[]): string[] {
  const warnings: string[] = [];
  if (items.some((item) => item.status === 'queued' || item.status === 'analyzing')) {
    warnings.push('Обработка ещё не завершена. Экспорт включает только завершённые снимки.');
  }
  const completed = items.filter((item) => item.status === 'ready' || item.status === 'error');
  if (completed.some((item) => item.status === 'error' && /\.zip$/i.test(item.filename))) {
    warnings.push('Ошибка архива: отчёт может содержать не все снимки. Исправьте архив и добавьте его повторно.');
  }
  const paths = completed.map((item) => item.inputPath ?? item.filename);
  if (new Set(paths).size !== paths.length) {
    warnings.push('В отчёте повторяются пути к снимкам. Для закрытого теста нужны различимые входные пути.');
  }
  if (completed.some((item) => /\.dcm$|\.dicom$/i.test(item.filename)
    && item.status === 'ready' && (!item.result?.study_uid || !item.result?.image_uid))) {
    warnings.push('У части обработанных DICOM отсутствуют Study UID или Image UID; проверьте исходные DICOM-теги.');
  }
  return warnings;
}

export function reportTable(items: WorkspaceItem[], kind: ReportKind = 'clinical'): { headers: string[]; rows: (string | number)[][] } {
  const completed = items.filter((item) => item.status === 'ready' || item.status === 'error');
  const withManual = kind === 'clinical';
  const headers = ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region', 'quality_class', 'violation_type', 'processing_status', 'time_of_processing'];
  if (withManual) headers.push('assessment_source', 'axis_top_x', 'axis_top_y', 'axis_bottom_x', 'axis_bottom_y', 'axis_angle_deg', 'axis_rule');
  const rows = completed.map((item) => {
    const r = kind === 'submission' ? item.result : resultWithSavedAxis(item);
    const success = r?.processing_status === 'Success';
    const row: (string | number)[] = [item.inputPath ?? item.filename, r?.study_uid ?? '', r?.image_uid ?? '', r?.anatomical_region ?? 'unknown',
      success ? r.quality_class ?? '' : '', success ? r.violation_types.join(';') : '', r?.processing_status ?? 'Failure', r?.time_of_processing ?? 0];
    if (withManual) {
      const manual = item.savedAxis && r !== item.result ? item.savedAxis : null;
      row.push(manual ? 'manual_axis_heuristic' : 'automatic', manual?.top[0] ?? '', manual?.top[1] ?? '',
        manual?.bottom[0] ?? '', manual?.bottom[1] ?? '', manual ? r?.geometry?.axis_line?.angle_deg ?? '' : '', manual ? 'angle_gt_5_deg' : '');
    }
    return row;
  });
  return { headers, rows };
}
