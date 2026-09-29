"""Compact hip training: one frozen public ResNet18 and two final LRs.

All labeled hips enter each final quality fit. Temporary study-disjoint fits
only calibrate the cutoff. Independent validation belongs to the separate
combined_qc.validation entry point, never to training-set measurements here.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from combined_qc.common import Progress, quiet_output, resolve_checkpoint_paths, write_json
from combined_qc.router.features import actual_source
from .bootstrap import OFFICIAL_SHA, SOURCE_NAME, prepare_backbone
from .features import (
    CALIBRATION_RECIPE, CLASSIFIER_NAMES, DATA, FEATURE_COUNTS, FEATURE_VIEWS, LR_PARAMS, MIN_RECALL,
    SEEDS, TASKS, crossfit_training_probabilities, extract_quality_features,
    feature_weights, fit_quality_classifier, positioning_calibration_context, read_quality_data,
    recipe_fingerprint, score_numeric, select_threshold, write_classifier,
)
from .preprocessing import sha256

_TRAIN_LOCK = threading.RLock()


def _data_signature(data):
    """Check native pixels too; a matching manifest alone cannot prove resume."""
    from combined_qc.router.features import load_record
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row['anatomical_region'] in {'left_hip', 'right_hip'}]
    hashes = {}
    for row in rows:
        load_record(row, data)
        hashes[row['image_id']] = sha256(actual_source(row, data))
    return {'manifest_sha256': sha256(data / 'manifest.jsonl'), 'splits_sha256': sha256(data / 'splits.json'),
            'source_files_sha256': hashes}


def _calibration_metrics(y, probability, threshold):
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from .preprocessing import centered_score
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {'n': len(y), 'positives': int(y.sum()), 'f1': float(f1_score(y, prediction, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, centered_score(probability, threshold))),
            'average_precision': float(average_precision_score(y, centered_score(probability, threshold))),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)), 'accuracy': float(np.mean(y == prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'scope': 'training_crossfit_cutoff_calibration; cutoff selected on these predictions; not independent validation'}


def train(data=None, *, checkpoints_dir=None, device='auto', epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    """Fit all final hip classifiers and optionally publish converted inference.

    The shared ImageNet encoder stays frozen; ``epochs`` is kept for the common
    module API and does not alter logistic training. ``resume=True`` only reuses
    complete fits bound to matching native data, public weights and fit code.
    No source preparation changes an existing runtime until conversion succeeds.
    """
    if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0):
        raise ValueError('epochs must be a positive integer or None')
    if str(device).lower() not in {'auto', 'cpu'}:
        raise ValueError('Compact hip training runs frozen ONNX features and logistic regression on CPU; use device="auto" or "cpu"')
    data = Path(data).expanduser().resolve() if data is not None else DATA
    base, source, runtime = resolve_checkpoint_paths('hip', checkpoints_dir)
    progress = Progress('hip', verbose or ci)
    with _TRAIN_LOCK, quiet_output(verbose or ci):
        source.mkdir(parents=True, exist_ok=True)
        try:
            return _train(data, base=base, source=source, runtime=runtime, progress=progress,
                          epochs=epochs, verbose=verbose, ci=ci, resume=resume, convert=convert)
        except BaseException as error:
            progress.update('training', f'Failed: {type(error).__name__}: {error}')
            raise


def _train(data, *, base, source, runtime, progress, epochs, verbose, ci, resume, convert):
    started = time.perf_counter()
    from .laterality.training import train as train_laterality
    from .laterality.conversion import export_runtime as export_side
    progress.update('training', 'Training/verifying automatic side classifier inside hip source')
    side_report = train_laterality(data, source_dir=source / 'laterality', verbose=verbose, ci=ci,
                                   resume=resume, convert=False)
    backbone = prepare_backbone(base, verbose=verbose, ci=ci)
    recipe = recipe_fingerprint()
    data_signature = _data_signature(data)
    quality = source / 'quality'
    report_path = base / 'training_report.json'
    resumed = False
    # Canonicalize using actual source side model, converted privately. Existing
    # live side weights are never selected implicitly during a fresh fit.
    with tempfile.TemporaryDirectory(dir=source, prefix='.side-for-quality-') as temporary:
        side_runtime = Path(temporary) / 'laterality'
        export_side(source / 'laterality', side_runtime, data=data, verbose=verbose, ci=ci, verify_samples=2)
        from .laterality.runtime import LateralityClassifier
        side = LateralityClassifier(runtime_dir=side_runtime)
        side_binding = {name: side.provenance[name] for name in ('backbone_sha256', 'classifier_sha256')}
        if resume and (quality / 'manifest.json').is_file() and report_path.is_file():
            from .convert_all import load_quality_source
            _, metadata, manifest = load_quality_source(source)
            report = json.loads(report_path.read_text())
            if (report.get('format') != 'hip_compact_training_v1'
                    or manifest.get('training_recipe_sha256') != recipe
                    or manifest.get('data_signature') != data_signature
                    or manifest.get('side_binding') != side_binding
                    or manifest.get('public_source_sha256') != sha256(source / SOURCE_NAME)):
                raise ValueError('Saved compact hip fit differs from current data, side model or code; use resume=False')
            expected_ids = [row['image_id'] for row in json.loads((source / 'quality/fit_membership.json').read_text())]
            if any(spec['fit_image_ids'] != expected_ids for spec in metadata.values()):
                raise ValueError('Saved compact quality classifiers have inconsistent fit membership')
            resumed = True
            progress.update('training', 'Reusing verified final single positioning and ROI classifiers', 2, 2)
        else:
            prepared = read_quality_data(data, side_runtime_dir=side_runtime, progress=progress)
            features = extract_quality_features(prepared, backbone, progress=progress,
                                                include_positioning_views=True)
            positioning_context = positioning_calibration_context(prepared, features,
                                                side_source_dir=source / 'laterality')
            mask, folds, groups = prepared['fit_mask'], prepared['folds'][prepared['fit_mask']], prepared['groups'][prepared['fit_mask']]
            fit_rows = [row for index, row in enumerate(prepared['rows']) if mask[index]]
            fit_ids, fit_studies = [row['image_id'] for row in fit_rows], sorted(set(groups))
            quality.mkdir(parents=True, exist_ok=True)
            models, calibration, predictions = {}, {}, {row['image_id']: {'image_id': row['image_id'], 'study_id': row['study_id'], 'fold': int(folds[index])}
                                                      for index, row in enumerate(fit_rows)}
            for task in TASKS:
                x, y = features[task][mask], prepared['targets'][task][mask]
                progress.update('training', f'{task}: calibrating cutoff with study-separated training crossfit')
                pooled, fold_reports = crossfit_training_probabilities(task, x, y, folds, groups,
                        progress=progress, positioning_context=positioning_context,
                        model_dir=source / 'outputs/cutoff_side')
                threshold = select_threshold(y, pooled, MIN_RECALL[task])
                progress.update('training', f'{task}: fitting one final classifier on all {len(y)} labeled hips')
                values = fit_quality_classifier(task, x, y)
                metadata = {'format': 'hip_compact_numeric_lr_source_v1', 'task': task,
                            'feature_view': FEATURE_VIEWS[task], 'feature_dimension': FEATURE_COUNTS[task],
                            'source_weights_sha256': OFFICIAL_SHA, 'fit_image_ids': fit_ids,
                            'fit_study_ids': fit_studies, 'training_n': len(y), 'positives': int(y.sum()),
                            'fit_dtype': 'float64', 'LR': LR_PARAMS, 'seed': SEEDS[task],
                            'feature_group_weights': feature_weights(task).tolist(),
                            'effective_scale': 'raw StandardScaler scale divided by feature_group_weights',
                            'threshold': threshold, 'minimum_recall': MIN_RECALL[task],
                            'threshold_source': 'Pooled five-fold cross-fitted training probabilities; positioning side preprocessing excludes every held study; never final-fit resubstitution',
                            'calibration_recipe': CALIBRATION_RECIPE,
                            'threshold_calibration': fold_reports, 'recipe_sha256': recipe,
                            'data_hashes': data_signature, 'pixel_hashes': prepared['provenance']['pixel_hashes'],
                            'side_binding': side_binding, 'independent_final_validation': False}
                filename = CLASSIFIER_NAMES[task]
                write_classifier(quality / filename, values, metadata, threshold)
                from .runtime import load_numeric_classifier
                restored, restored_spec = load_numeric_classifier(quality / filename, task, expected_metadata=metadata)
                error = float(np.max(np.abs(score_numeric(values, x) - score_numeric(restored, x))))
                if error > 1e-12:
                    raise AssertionError('Saved hip source LR numeric roundtrip differs')
                models[task] = metadata
                calibration[task] = {'metrics': _calibration_metrics(y, pooled, threshold), 'threshold': threshold,
                                     'folds': fold_reports, 'numeric_roundtrip_max_error': error}
                for index, row in enumerate(fit_rows):
                    predictions[row['image_id']][task] = {'truth': int(y[index]), 'probability': float(pooled[index])}
            write_json(quality / 'fit_membership.json', [{'image_id': row['image_id'], 'study_id': row['study_id']} for row in fit_rows])
            manifest = {'format': 'hip_compact_quality_source_v1', 'public_source_sha256': OFFICIAL_SHA,
                        'files': {filename: sha256(quality / filename) for filename in CLASSIFIER_NAMES.values()},
                        'models': models, 'side_binding': side_binding,
                        'training_recipe_sha256': recipe, 'data_signature': data_signature,
                        'public_encoder_fine_tuned_on_dxa': False,
                        'quality_models': {'neural': 1, 'ml': 2}, 'final_quality_fit_n': len(fit_rows)}
            write_json(quality / 'manifest.json', manifest)
            write_json(source / 'outputs/cutoff_calibration_predictions.json', list(predictions.values()))
            write_json(source / 'outputs/cutoff_calibration.json', calibration)
            report = {'format': 'hip_compact_training_v1', 'region': 'hip',
                      'checkpoints_dir': str(base), 'source_dir': str(source), 'runtime_dir': str(runtime),
                      'models': {'quality_frozen_resnet18': 1, 'quality_logistic_regressions': 2,
                                 'laterality_frozen_resnet18': 1, 'laterality_classifier': 1},
                      'final_quality_fit_n': len(fit_rows), 'quality_fit_studies': len(fit_studies),
                      'frozen_encoder_dataset': 'ImageNet', 'neural_fine_tuning': False,
                      'device': 'cpu', 'epochs': None, 'epochs_argument': epochs,
                      'training_recipe_sha256': recipe, 'data_signature': data_signature,
                      'calibration_recipe': CALIBRATION_RECIPE,
                      'cutoff_training_calibration': calibration,
                      'validation_metrics': None,
                      'validation_note': 'Run the separate grouped validation script. Calibration metrics select final cutoff on training data and must not be reported as independent validation.',
                      'metrics_path': str(source / 'outputs/cutoff_calibration.json'),
                      'predictions_path': str(source / 'outputs/cutoff_calibration_predictions.json'),
                      'laterality': side_report}
    report.update(resumed=resumed, converted=False, conversion=None, seconds=time.perf_counter() - started)
    write_json(report_path, report)
    if convert:
        from .convert_all import convert_all
        report['conversion'] = convert_all(base, data=data, device='cpu', verbose=verbose, ci=ci)
        report['converted'] = True
        write_json(report_path, report)
    progress.update('training', 'Complete: one shared quality encoder, two final classifiers and automatic side', 3, 3)
    return report
