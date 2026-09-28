"""Complete tables from saved excluded-study predictions, without fitting.

Fixed spine rules and ORs containing them retain the development-data caveat.
The routed table accounts explicitly for unavailable/needs-review decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json

ROOT = Path(__file__).resolve().parent


def build_tables(predictions, rows, spine_rows):
    from .validation import LEARNED_SCORE_TYPES, _metrics, learned_score_diagnostics
    by_id = {r['image_id']: r for r in rows}
    spine = {r['image_id']: r for r in spine_rows}
    if len(predictions) != len(by_id) or {p['image_id'] for p in predictions} != set(by_id):
        raise ValueError('Predictions must cover every dataset image exactly once')
    names = {
        'hip_positioning_rotation': 'Таз: укладка / ротация', 'hip_roi': 'Таз: ROI',
        'spine_positioning': 'Позвоночник: охват', 'spine_axis': 'Позвоночник: угол',
        'spine_artifacts': 'Позвоночник: артефакты',
    }
    observations = {key: [] for key in (*names, 'hip_or', 'spine_or', 'overall_or', 'region', 'side')}
    routed = {key: [] for key in ('hip_or', 'spine_or', 'overall_or')}
    unavailable = {key: [] for key in routed}
    expert_total = {key: [] for key in routed}
    routed_expert_total = {key: [] for key in routed}
    total_disagreements = []
    for p in predictions:
        row = by_id[p['image_id']]
        if p['global_study_id'] != row['study_id'] or not p['study_excluded_from_all_prediction_model_fits']:
            raise ValueError('Prediction study provenance is inconsistent')
        region = int(row['anatomical_region'] == 'lumbar_spine')
        if p['region_truth'] != region:
            raise ValueError('Saved region truth disagrees with dataset labels')
        observations['region'].append((region, p['region_prediction'], p['region_probability_spine']))
        is_hip = region == 0
        if is_hip:
            side = int(row['anatomical_region'] == 'right_hip')
            if p['side_truth'] != side:
                raise ValueError('Saved side truth disagrees with dataset labels')
            observations['side'].append((side, int(p['automatic_side'] == 'right_hip'), p['side_probability_right']))
            entries = {key: p.get('hip', {}).get(short) for key, short in
                       (('hip_positioning_rotation', 'positioning_rotation'), ('hip_roi', 'roi'))}
            truth = row
        else:
            truth = spine[row['image_id']]
            fixed = p.get('development_only_fixed_rules', {})
            entries = {'spine_positioning': fixed.get('coverage'),
                       'spine_axis': fixed.get('axis'), 'spine_artifacts': p.get('artifacts')}
        targets, decisions = [], []
        for task, result in entries.items():
            target = truth['targets'][task]
            if target is None:
                continue
            if result is None:
                raise ValueError(f"Missing saved task prediction: {p['image_id']}/{task}; run validation with fixed rules")
            prediction = result.get('prediction', result.get('violation'))
            if prediction is None:
                raise ValueError('Unavailable task prediction must not be silently excluded')
            if task == 'spine_artifacts' and result['truth'] != target:
                raise ValueError('Artifact truth disagrees with corrected labels')
            if task.startswith('hip_') and result['truth'] != target:
                raise ValueError('Hip truth disagrees with dataset labels')
            score = result.get('score', result.get('probability'))
            observations[task].append((int(target), int(prediction), float(score)))
            targets.append(int(target)); decisions.append(int(prediction))
        if not targets:
            continue
        if len(targets) != len(entries) or bool(truth['quality_label']) != any(targets):
            raise ValueError('Incomplete task labels or inconsistent OR truth')
        value = (int(any(targets)), int(any(decisions)), None)
        branch = 'hip_or' if is_hip else 'spine_or'
        observations[branch].append(value); observations['overall_or'].append(value)
        available = p['region_prediction'] == region and not p['region_needs_review']
        for key in (branch, 'overall_or'):
            if available:
                routed[key].append(value)
            else:
                unavailable[key].append(p['image_id'])
        # The main reference remains the accepted OR of criterion labels.
        # Spine manifests also preserve the original expert total column.
        original_total = truth['quality_label'] if is_hip else truth.get('source_excel_total')
        if original_total is not None:
            if original_total not in (0, 1):
                raise ValueError('Original expert total must be binary or unavailable')
            original_value = (int(original_total), value[1], None)
            for key in (branch, 'overall_or'):
                expert_total[key].append(original_value)
                if available:
                    routed_expert_total[key].append(original_value)
            if int(original_total) != value[0]:
                total_disagreements.append({'image_id': p['image_id'], 'original_expert_total': int(original_total),
                                            'criterion_or': value[0]})

    def metric(values, macro=False):
        y, pred, scores = zip(*values)
        return _metrics(y, pred, scores if all(s is not None for s in scores) else None, macro=macro)

    table = []
    diagnostics = learned_score_diagnostics(predictions)
    order = [('region', 'Область: таз / позвоночник (macro F1)'),
             ('side', 'Сторона таза (macro F1)'), *names.items(),
             ('hip_or', 'Таз: OR'), ('spine_or', 'Позвоночник: OR'), ('overall_or', 'Все профили: OR')]
    for key, title in order:
        mixed = key in {'spine_positioning', 'spine_axis', 'spine_or', 'overall_or'}
        fixed_only = key in {'spine_positioning', 'spine_axis'}
        macro = key in {'region', 'side'}
        metrics = metric(observations[key], macro)
        positive_class = {'region': 'lumbar_spine', 'side': 'right_hip_visual_orientation'}.get(key, 'quality_violation')
        scope = ('Development dataset evaluation of previously selected fixed rules' if fixed_only else
                 'Internal excluded-study ML OOF combined with development-data fixed rules' if mixed else
                 'Internal excluded-study OOF of a previously selected training recipe')
        table.append({'key': key, 'name': title, 'metrics': metrics,
                      'f1_averaging': 'macro' if macro else 'binary',
                      'reported_f1_key': 'macro_f1' if macro else 'f1',
                      'precision_recall_averaging': 'binary', 'positive_class': positive_class,
                      'class_counts': {'positive': metrics['positives'], 'negative': metrics['n'] - metrics['positives']},
                      'roc_auc_score_type': LEARNED_SCORE_TYPES.get(key, 'fixed_rule_operational_score' if fixed_only else None),
                      'score_diagnostics': diagnostics.get(key), 'evaluation_scope': scope,
                      'contains_development_rules': mixed,
                      'scope': scope})
    return {'format': 'complete_qc_fold_tables_v1', 'n_processed': len(predictions),
            'main_quality_reference': 'OR of labelled task criteria; corrected local spine criteria override global spine labels',
            'metric_interpretation': 'Internal exploratory recipe estimates, not independent test scores of the shipped all-data checkpoints',
            'learned_score_diagnostics': diagnostics,
            'profile_tables': table,
            'routed_or': {key: {'n_labeled': len(observations[key]), 'n_available': len(routed[key]),
                               'unavailable_ids': unavailable[key],
                               'coverage': len(routed[key]) / len(observations[key]),
                               'metrics_on_available_only': metric(routed[key]) if routed[key] else None,
                               'evaluation_scope': 'Conditional on region-correct, region-confidence-accepted decisions; not all labelled images',
                               'f1_averaging': 'binary', 'positive_class': 'any_quality_violation',
                               'region_confidence_cutoff': .9,
                               'contains_development_rules': key != 'hip_or'} for key in routed},
            'alternative_original_expert_total': {
                'changes_main_reference': False,
                'reference': 'Hip global quality_label; spine preserved source_excel_total',
                'different_reference_image_ids': total_disagreements,
                'profile_metrics': {key: metric(values) if values else None for key, values in expert_total.items()},
                'routed_metrics_on_available_only': {key: metric(values) if values else None
                                                     for key, values in routed_expert_total.items()},
                'comparison_requires_same_n': {key: len(expert_total[key]) == len(observations[key]) for key in routed}},
            'limitations': ['Spine coverage and angle rules were developed on these data; their rows and related ORs are not independent validation',
                            'Profile tables evaluate each appropriate module before router confidence rejection',
                            'Routed metrics use available decisions only and must be read together with coverage',
                            'Recipes were previously selected on these data; this is not a sealed independent test',
                            'Different tasks, class prevalences and macro/binary F1 are not a ranking of model superiority',
                            'Hip operational scores use fold-specific thresholds; pooled score AUC and raw-probability AUC can differ',
                            'Side needs_review flags do not suppress QC labels in the current runtime; region rejection does',
                            'Seven positive ROI images in six studies make point estimates unstable; no independent-test confidence interval is claimed']}


def markdown(report):
    headers = '| Проверка | N | Положительных | F1 | ROC AUC | Accuracy | Balanced accuracy | Precision | Recall |'
    separator = '|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    lines = ['# Полная таблица проверки по валидационным фолдам', '',
             'Обучаемые классификаторы исключают проверяемое исследование. Замороженные нейросети и фиксированные правила запускаются без переобучения. '
             'Строки со звёздочкой содержат правила охвата/угла, разработанные на этих данных, и не являются независимой валидацией.', '', headers, separator]
    for row in report['profile_tables']:
        m = row['metrics']; macro = row['key'] in {'region', 'side'}
        values = [m.get(k) for k in ('macro_f1' if macro else 'f1', 'roc_auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall')]
        formatted = ['—' if v is None else f'{v:.4f}' for v in values]
        title = row['name'] + (' *' if row['contains_development_rules'] else '')
        lines.append(f"| {title} | {m['n']} | {m['positives']} | " + ' | '.join(formatted) + ' |')
    lines += ['', 'F1/precision/recall относятся к нарушениям качества; у области и стороны F1 — macro, а precision/recall — для позвоночника и правой стороны соответственно. '
              'ROC AUC у OR не рассчитывается: рабочий OR возвращает только метку. '
              'Значения разных задач и разные варианты усреднения F1 нельзя использовать для ранжирования моделей.', '',
              '## Какая шкала используется для ROC AUC', '',
              'Основная таблица сохраняет ROC AUC рабочего score. У таза это score относительно порога своего фолда; '
              'разные пороги могут менять общий порядок снимков между фолдами. '
              'Поэтому ниже отдельно показана AUC исходной вероятности LR на тех же внешних предсказаниях. '
              'У артефактов probability уже включает поправку по доле положительных случаев в обучении. '
              'Значения каждого фолда доступны в `learned_score_diagnostics` JSON.', '',
              '| Обучаемая модель | Pooled AUC рабочего score | Pooled AUC model probability |',
              '|---|---:|---:|']
    titles = {row['key']: row['name'] for row in report['profile_tables']}
    for key, diagnostic in report['learned_score_diagnostics'].items():
        vals = ['—' if diagnostic[field] is None else f"{diagnostic[field]:.4f}"
                for field in ('roc_auc_operational_score', 'roc_auc_probability')]
        lines.append(f"| {titles[key]} | " + ' | '.join(vals) + ' |')
    lines += ['',
              '## Общий запуск с порогом уверенности маршрутизатора', '',
              '| Результат | Решений / с разметкой | Покрытие | F1 доступных решений | Accuracy доступных решений | Balanced accuracy доступных решений |',
              '|---|---:|---:|---:|---:|---:|']
    for key, title in (('hip_or', 'Таз: OR'), ('spine_or', 'Позвоночник: OR *'), ('overall_or', 'Весь пайплайн: OR *')):
        row = report['routed_or'][key]; m = row['metrics_on_available_only']
        vals = ['—' if m is None else f"{m[k]:.4f}" for k in ('f1', 'accuracy', 'balanced_accuracy')]
        lines.append(f"| {title} | {row['n_available']}/{row['n_labeled']} | {row['coverage']:.4f} | " + ' | '.join(vals) + ' |')
    lines += ['', 'При отказе маршрутизатора области решения качества нет. Такие снимки явно отражены в покрытии и `unavailable_ids`; '
              'числа второй таблицы условны на доступных решениях, а не на всём наборе. '
              'Например, F1 для 248/249 — оценка 248 доступных решений, а не полный F1 249 снимков. '
              'Здесь отказ означает отказ маршрутизатора области; флаг низкой уверенности стороны в текущем runtime сохраняет метки качества. '
              'Звёздочка сохраняет то же ограничение правил позвоночника.', '',
              'Источник: `predictions.json` из того же прогона; ни одно значение финальных классификаторов, обученных на всех метках, сюда не подмешивается.', '']
    alternative = report['alternative_original_expert_total']
    lines += ['## Альтернативный эталон исходной итоговой колонки', '',
              f"Основной эталон остаётся OR критериев. Исходный экспертный итог отличается у {len(alternative['different_reference_image_ids'])} снимков позвоночника. "
              'Ниже меняется только эталон; используются те же сохранённые предсказания. Сопоставление требует одинакового N. '
              'Это разные цели оценки, и ни одна из них здесь не объявляется официальным протоколом хакатона.', '',
              '| Эталон и состав | N | Положительных | F1 |', '|---|---:|---:|---:|']
    main = next(row['metrics'] for row in report['profile_tables'] if row['key'] == 'overall_or')
    for title, m in [('OR критериев, все профили', main),
                     ('Исходный экспертный итог, все профили', alternative['profile_metrics']['overall_or']),
                     ('OR критериев, доступные решения', report['routed_or']['overall_or']['metrics_on_available_only']),
                     ('Исходный экспертный итог, доступные решения', alternative['routed_metrics_on_available_only']['overall_or'])]:
        if m is not None:
            lines.append(f"| {title} | {m['n']} | {m['positives']} | {m['f1']:.4f} |")
    lines += ['', 'Это внутренняя исследовательская проверка уже выбранных рецептов. '
              'Её можно приводить вместе с финальными моделями, обученными на всех данных, как оценку процедуры обучения; '
              'она не является независимой проверкой именно поставляемых весов. '
              'Для ROI доступны только семь положительных снимков из шести исследований, поэтому точечный результат нестабилен.', '']
    return '\n'.join(lines)


def refresh_saved_metrics_report(output_dir, tables):
    """Refresh report semantics from saved predictions; do not fit or infer."""
    from .validation import annotate_grouped_metrics, _report_markdown
    output = Path(output_dir).resolve()
    path = output / 'metrics.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    provenance = saved['provenance']
    if (provenance['manifest_sha256'] != tables['provenance']['manifest_sha256']
            or provenance['corrected_spine_labels_sha256'] != tables['provenance']['spine_labels_sha256']):
        raise ValueError('Saved validation report and regenerated tables refer to different labels')
    previous_prediction_hash = saved.get('complete_tables', {}).get('provenance', {}).get('predictions_sha256')
    if previous_prediction_hash is not None and previous_prediction_hash != tables['provenance']['predictions_sha256']:
        raise ValueError('Saved validation predictions changed; report refresh must not relabel a different run')
    # Refuse to label unrelated predictions as the source of an older run.
    bindings = {'region': saved['grouped_validation']['region_classifier'],
                'side': saved['grouped_validation']['hip_laterality'],
                'hip_positioning_rotation': saved['grouped_validation']['hip']['positioning_rotation'],
                'hip_roi': saved['grouped_validation']['hip']['roi'],
                'hip_or': saved['grouped_validation']['hip']['or'],
                'spine_artifacts': saved['grouped_validation']['spine']['artifacts']}
    for row in tables['profile_tables']:
        if row['key'] not in bindings:
            continue
        previous = bindings[row['key']]
        for key, value in row['metrics'].items():
            if key not in previous:
                continue
            old = previous[key]
            if (value is None) != (old is None) or (value is not None and abs(float(old) - float(value)) > 1e-12):
                raise ValueError(f"Saved predictions no longer reproduce {row['key']}/{key}; run validation again")
    predictions = json.loads((output / 'predictions.json').read_text(encoding='utf-8'))
    annotate_grouped_metrics(saved['grouped_validation'], predictions)
    saved['complete_tables'] = tables
    saved['report_only_regeneration'] = {'source': 'Existing saved outer-fold predictions; no fitting or neural inference',
                                       'predictions_sha256': tables['provenance']['predictions_sha256'],
                                       'training_performed': False, 'original_run_provenance_preserved': True}
    write_json(path, saved)
    (output / 'README.md').write_text(_report_markdown(saved), encoding='utf-8')
    return saved


def write_tables(output_dir, *, data=None, spine_data=None, refresh_report=False):
    output = Path(output_dir).resolve()
    data = Path(data).resolve() if data is not None else ROOT / 'data/dxa_v1'
    spine_data = Path(spine_data).resolve() if spine_data is not None else ROOT / 'spine/data'
    read_rows = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prediction_path = output / 'predictions.json'
    report = build_tables(json.loads(prediction_path.read_text()), read_rows(data / 'manifest.jsonl'),
                          read_rows(spine_data / 'labeled_manifest.jsonl'))
    from .router.features import sha256
    report['provenance'] = {'predictions_sha256': sha256(prediction_path),
                            'manifest_sha256': sha256(data / 'manifest.jsonl'),
                            'spine_labels_sha256': sha256(spine_data / 'labeled_manifest.jsonl')}
    write_json(output / 'complete_metrics.json', report)
    (output / 'COMPLETE_TABLES.md').write_text(markdown(report), encoding='utf-8')
    if refresh_report:
        refresh_saved_metrics_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/cross_validation')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--spine-data', type=Path)
    parser.add_argument('--refresh-report', action='store_true',
                        help='Also refresh metrics.json/README from saved predictions; never train or infer')
    args = parser.parse_args()
    write_tables(args.output_dir, data=args.data_dir, spine_data=args.spine_data,
                 refresh_report=args.refresh_report)


if __name__ == '__main__':
    main()
