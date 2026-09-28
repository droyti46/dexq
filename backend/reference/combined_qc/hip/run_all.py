"""Run the compact deployment on every hip image (descriptive metrics only).

Independent excluded-study validation: python -m combined_qc.validate --ci.
This command never extracts old ensemble folds or treats training scores as
validation. Optional --train explicitly fits and converts the final models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..common import Progress, write_json
from ..report_metrics import _metrics
from ..router.features import actual_source, load_record
from .pipeline import DATA_ROOT, HIP_TASKS, PROJECT_ROOT, infer, load_checkpoints

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs/all'


def run_all(data=DEFAULT_DATA_ROOT, checkpoints_dir=None, output_dir=DEFAULT_OUTPUT_DIR,
            *, device='auto', verbose=False, ci=False, train=False):
    data_root, output_dir = Path(data).resolve(), Path(output_dir).resolve()
    manifest = data_root / 'manifest.jsonl'
    rows = sorted((json.loads(line) for line in manifest.read_text().splitlines() if line.strip()),
                  key=lambda row: row['image_id'])
    rows = [row for row in rows if row['anatomical_region'] in {'left_hip', 'right_hip'}]
    if len(rows) != 155 or len({row['image_id'] for row in rows}) != 155:
        raise ValueError('Expected all 155 unique hip images; no input may be excluded')
    if train:
        from .pipeline import train as train_models
        train_models(data_root, checkpoints_dir=checkpoints_dir, device=device,
                     verbose=verbose, ci=ci, resume=False, convert=True)
    model = load_checkpoints(checkpoints_dir, device=device, verbose=verbose, ci=ci)
    progress = Progress('hip', enabled=bool(verbose or ci))
    results, evaluation = [], []
    for index, row in enumerate(rows, 1):
        # Verify native pixels and use the public file inference, including side.
        load_record(row, data_root, PROJECT_ROOT / 'source_data')
        path = actual_source(row, data_root, PROJECT_ROOT / 'source_data')
        result = infer(path, model=model, device=device)
        result['metadata'].update(image_id=row['image_id'], study_id=row['study_id'])
        if not isinstance(result.get('annotated_image'), dict):
            raise ValueError(f"Missing ready visualization: {row['image_id']}")
        results.append(result)
        for task in HIP_TASKS:
            evaluation.append({'image_id': row['image_id'], 'task': task,
                               'true': row['targets'].get(task),
                               'prediction': result['labels'].get(task),
                               'score': result['scores'].get(task)})
        progress.update('infer', row['image_id'], index, len(rows))
    metrics = {task: _metrics([item for item in evaluation if item['task'] == task],
                             truth_key='true', prediction_key='prediction', score_key='score',
                             score_semantics='single LR threshold-centered uncalibrated score')
               for task in HIP_TASKS}
    pooled = [{'image_id': row['image_id'], 'true': row.get('quality_label'),
               'prediction': result['any_violation']} for row, result in zip(rows, results)]
    metrics['hip_or'] = _metrics(pooled, truth_key='true', prediction_key='prediction')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'n_processed': len(results), 'inference_runtime': 'onnx_and_numeric_npz',
              'metrics_scope': 'Descriptive deployment inference, includes the final training images; NOT validation',
              'validation_command': 'python -m combined_qc.validate --ci', 'metrics': metrics,
              'provenance': {'data_root': str(data_root), 'checkpoints_dir': str(model.checkpoints_dir),
                             'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    write_json(output_dir / 'predictions_full.json', results)
    write_json(output_dir / 'metrics.json', report)
    lines = ['# Таз: полный рабочий инференс', '',
             'Все 155 снимков обработаны; метрики качества — по 150 размеченным.',
             '**Это описательный прогон, включающий обучение, а не валидация.**', '',
             '| Подзадача | F1 | ROC AUC | Balanced accuracy |', '|---|---:|---:|---:|']
    for task, value in metrics.items():
        auc = '—' if value.get('roc_auc') is None else f"{value['roc_auc']:.4f}"
        lines.append(f"| {task} | {value['f1']:.4f} | {auc} | {value['balanced_accuracy']:.4f} |")
    lines.extend(['', 'Для валидации: `python -m combined_qc.validate --ci`.'])
    (output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', '--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--checkpoints-dir', '--checkpoint-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', choices=('auto', 'cpu'), default='auto')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ci', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    run_all(args.data, args.checkpoints_dir, args.output_dir, device=args.device,
            verbose=args.verbose, ci=args.ci, train=args.train)


if __name__ == '__main__':
    main()
