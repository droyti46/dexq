"""Shared native-pixel features and numeric logistic fits for hip quality.

The encoder is the pinned, frozen public ImageNet ResNet18. Positioning uses
its stage3 5x5 grid in automatic canonical orientation; ROI uses its global
pool in native orientation plus seven acquisition geometry values.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np

from combined_qc.common import Progress
from combined_qc.router.features import actual_source, load_record
from .preprocessing import acquisition_features, centered_score, letterbox, sha256

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data/dxa_v1'
TASKS = ('positioning_rotation', 'roi')
FEATURE_COUNTS = {'positioning_rotation': 6400, 'roi': 519}
FEATURE_VIEWS = {'positioning_rotation': 'stage3_6400', 'roi': 'global512_acquisition7'}
SEEDS = {'positioning_rotation': 20260928, 'roi': 20260927}
MIN_RECALL = {'positioning_rotation': .70, 'roi': .50}
LR_PARAMS = {'C': .01, 'class_weight': 'balanced', 'solver': 'liblinear', 'tol': 1e-6, 'max_iter': 3000}
CLASSIFIER_NAMES = {'positioning_rotation': 'positioning_logreg.npz', 'roi': 'roi_logreg.npz'}
CALIBRATION_RECIPE = 'study_excluded_side_then_quality_crossfit_v2'


def recipe_fingerprint():
    """Bind all preparation/fit/export code and numerical library versions."""
    names = ('features.py', 'training.py', 'bootstrap.py', 'onnx_models.py', 'preprocessing.py')
    from .bootstrap import export_recipe_sha256
    value = {'encoder_export_recipe_sha256': export_recipe_sha256(), 'code': {name: sha256(ROOT / name) for name in names},
             'libraries': {name: importlib.metadata.version(name) for name in
                           ('numpy', 'Pillow', 'scikit-learn', 'onnx', 'onnxruntime', 'torch', 'torchvision')},
             'LR': LR_PARAMS, 'seeds': SEEDS, 'minimum_recall': MIN_RECALL,
             'feature_views': FEATURE_VIEWS, 'fit_dtype': 'float64',
             'calibration_recipe': CALIBRATION_RECIPE,
             'laterality_fit_code_sha256': sha256(ROOT / 'laterality/training.py')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assign_study_folds(rows, split):
    """Extend saved folds to all studies without consulting targets or scores.

    Auxiliary side-only hips must follow their entire study, including studies
    that also contain spine images. This is the global validation assignment.
    """
    by_id = {row['image_id']: row for row in rows}
    if len(by_id) != len(rows) or set(split['folds']) != {str(i) for i in range(5)}:
        raise ValueError('Expected unique images and five saved global study folds')
    assigned = {}
    for key, spec in split['folds'].items():
        for image_id in spec['val']:
            if image_id not in by_id:
                raise ValueError('Saved validation image is absent from the manifest')
            study, fold = by_id[image_id]['study_id'], int(key)
            if study in assigned and assigned[study] != fold:
                raise ValueError('A study crosses saved validation folds')
            assigned[study] = fold
    counts = np.bincount(list(assigned.values()), minlength=5)
    for study in sorted({row['study_id'] for row in rows} - assigned.keys()):
        fold = int(np.argmin(counts))
        assigned[study] = fold
        counts[fold] += 1
    return np.asarray([assigned[row['study_id']] for row in rows], dtype=int)


def read_quality_data(data=DATA, *, side_runtime_dir=None, side_predictions=None, progress=None):
    """Load all native hips and verify pixel hashes, folds and automatic side.

    ``side_predictions`` may supply independently cross-fitted automatic side
    predictions for validation. Source anatomical labels are never used to
    canonicalize positioning inputs.
    """
    data = Path(data).resolve()
    progress = progress or Progress('hip', False)
    manifest_path, split_path = data / 'manifest.jsonl', data / 'splits.json'
    all_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    rows = [row for row in all_rows if row['anatomical_region'] in {'left_hip', 'right_hip'}]
    if not rows or len({row['image_id'] for row in rows}) != len(rows):
        raise ValueError('Hip manifest is empty or contains duplicate image IDs')
    split = json.loads(split_path.read_text())
    if split['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Hip folds do not match the manifest')
    all_folds = dict(zip((row['image_id'] for row in all_rows), assign_study_folds(all_rows, split)))
    specs = split['folds']
    if isinstance(specs, dict):
        items = [(int(key), value) for key, value in specs.items()]
    elif isinstance(specs, list):
        items = [(int(value.get('fold', index)), value) for index, value in enumerate(specs)]
    else:
        raise ValueError('Unexpected hip fold specification')
    lookup = {}
    for fold, spec in items:
        for image_id in spec.get('val', spec.get('val_image_ids', [])):
            if image_id in lookup:
                raise ValueError(f'Duplicate validation fold for {image_id}')
            lookup[image_id] = fold
    fit_mask = np.asarray([bool(row['eligible'] and row['source_split'] == 'train') for row in rows])
    folds = np.asarray([lookup.get(row['image_id'], -1) if fit_mask[index] else -1
                        for index, row in enumerate(rows)], dtype=int)
    groups = np.asarray([row['study_id'] for row in rows])
    if not fit_mask.any() or set(folds[fit_mask]) != set(range(5)):
        raise ValueError('Labeled hips need exactly five complete validation folds')
    targets = {task: np.asarray([int(row['targets']['hip_' + task]) if fit_mask[index] else -1
                                for index, row in enumerate(rows)], dtype=int) for task in TASKS}
    for task, y in targets.items():
        if not set(y[fit_mask]).issubset({0, 1}) or len(np.unique(y[fit_mask])) != 2:
            raise ValueError(f'{task}: both binary quality classes are required')
    for outer in range(5):
        train, test = fit_mask & (folds != outer), fit_mask & (folds == outer)
        if set(groups[train]) & set(groups[test]):
            raise ValueError(f'Study overlap in hip fold {outer}')
    side = None
    if side_predictions is None:
        from .laterality.runtime import LateralityClassifier
        side = LateralityClassifier(runtime_dir=side_runtime_dir)
    elif set(side_predictions) != {row['image_id'] for row in rows}:
        raise ValueError('Cross-fitted side predictions must cover all native hips exactly')
    native_frames, canonical_frames, acquisition, sides, source_hashes = [], [], [], [], {}
    for index, row in enumerate(rows):
        pixels = load_record(row, data)
        source_hashes[row['image_id']] = sha256(actual_source(row, data))
        predicted_side = (side_predictions[row['image_id']] if side_predictions is not None
                          else side.predict_array(pixels)['laterality'])
        if predicted_side not in {'left_hip', 'right_hip'}:
            raise ValueError(f"Automatic side unavailable for {row['image_id']}")
        frame, geometry = letterbox(pixels)
        if any(geometry[key] != row[key] for key in geometry):
            raise ValueError(f"Native/prepared geometry differs for {row['image_id']}")
        frame = np.rint(frame * 255).astype(np.uint8)
        native_frames.append(frame)
        canonical_frames.append(frame[:, ::-1].copy() if predicted_side == 'right_hip' else frame)
        acquisition.append(acquisition_features(geometry))
        sides.append(predicted_side)
        progress.update('prepare', 'Verified native pixels and automatic side', index + 1, len(rows))
    provenance = {'manifest_sha256': sha256(manifest_path), 'splits_sha256': sha256(split_path),
                  'pixel_hashes': {row['image_id']: row['pixel_hash'] for row in rows},
                  'source_files_sha256': source_hashes,
                  'automatic_side': 'cross_fitted' if side_predictions is not None else 'final_source_classifier',
                  'side_matches_manifest': bool(np.array_equal(sides, [row['anatomical_region'] for row in rows])),
                  'patient_independence_verified': bool(split.get('patient_independence_verified', False))}
    if side is not None:
        provenance['side_binding'] = {key: side.provenance[key] for key in ('backbone_sha256', 'classifier_sha256')}
    return {'rows': rows, 'fit_mask': fit_mask, 'folds': folds, 'groups': groups, 'targets': targets,
            'side_folds': np.asarray([all_folds[row['image_id']] for row in rows]),
            'native_frames': np.asarray(native_frames), 'canonical_frames': np.asarray(canonical_frames),
            'acquisition': np.asarray(acquisition, dtype=np.float64), 'sides': np.asarray(sides),
            'provenance': provenance, 'data_dir': data}


def extract_quality_features(prepared_data, backbone_path, *, progress=None, batch_size=16,
                             include_positioning_views=False):
    """Use the actual shared ONNX graph for both training and validation."""
    import onnxruntime as ort
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.log_severity_level = 3
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    session = ort.InferenceSession(str(backbone_path), sess_options=options, providers=['CPUExecutionProvider'])
    if [(value.name, value.type) for value in session.get_inputs()] != [('gray_u8', 'tensor(uint8)')]:
        raise ValueError('Unexpected shared hip feature input')
    if [value.name for value in session.get_outputs()] != ['global_512', 'spatial_2048', 'stage3_6400']:
        raise ValueError('Unexpected shared hip feature outputs')
    progress = progress or Progress('hip', False)
    def extract(frames, names, label):
        if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (320, 320):
            raise ValueError('Expected uint8 [N,320,320] prepared hips')
        parts = {name: [] for name in names}
        for start in range(0, len(frames), batch_size):
            values = session.run(list(names), {'gray_u8': np.ascontiguousarray(frames[start:start + batch_size])})
            for name, value in zip(names, values):
                parts[name].append(value)
            progress.update('features', f'Frozen shared ResNet18: {label}', min(start + batch_size, len(frames)), len(frames))
        return {name: np.concatenate(values).astype(np.float64) for name, values in parts.items()}

    if include_positioning_views:
        normal = extract(prepared_data['native_frames'], ('global_512', 'stage3_6400'), 'native calibration view')
        mirrored = extract(prepared_data['native_frames'][:, :, ::-1], ('stage3_6400',), 'mirrored calibration view')
        results = {'positioning_native': normal['stage3_6400'],
                   'positioning_mirrored': mirrored['stage3_6400']}
        results['positioning_rotation'] = np.where(prepared_data['sides'][:, None] == 'right_hip',
                results['positioning_mirrored'], results['positioning_native'])
        roi = normal['global_512']
    else:
        results = {'positioning_rotation': extract(prepared_data['canonical_frames'],
                               ('stage3_6400',), 'positioning_rotation')['stage3_6400']}
        roi = extract(prepared_data['native_frames'], ('global_512',), 'roi')['global_512']
    results['roi'] = np.column_stack([roi, prepared_data['acquisition']])
    for task in TASKS:
        x = results[task]
        frames = prepared_data['native_frames']
        if x.shape != (len(frames), FEATURE_COUNTS[task]) or not np.isfinite(x).all():
            raise ValueError(f'{task}: malformed extracted features')
    if include_positioning_views and any(results[key].shape != results['positioning_rotation'].shape
            or not np.isfinite(results[key]).all() for key in ('positioning_native', 'positioning_mirrored')):
        raise ValueError('Malformed positioning calibration feature views')
    return results


def positioning_calibration_context(prepared, features, *, side_source_dir):
    """Use the verified frozen side cache, never the final side LR predictions."""
    source = Path(side_source_dir)
    metadata = json.loads((source / 'features.json').read_text())
    fingerprint = metadata['fingerprint']
    cache = source / 'features.npz'
    if (metadata.get('cache_sha256') != sha256(cache)
            or fingerprint.get('manifest_sha256') != prepared['provenance']['manifest_sha256']
            or fingerprint.get('source_files_sha256') != prepared['provenance']['source_files_sha256']
            or fingerprint.get('backbone_sha256') != sha256(source / 'features_backbone.onnx')):
        raise ValueError('Side calibration feature cache does not match verified quality data')
    ids = np.asarray([row['image_id'] for row in prepared['rows']])
    with np.load(cache, allow_pickle=False) as saved:
        side_features = saved['features'].copy()
        if not np.array_equal(saved['image_ids'], ids):
            raise ValueError('Side calibration feature/image order differs from quality data')
    if side_features.shape != (len(ids), 2560) or not np.isfinite(side_features).all():
        raise ValueError('Malformed frozen side calibration features')
    mask = prepared['fit_mask']
    return {'normal': features['positioning_native'][mask],
            'mirrored': features['positioning_mirrored'][mask],
            'side_features': side_features[:, 512:].astype(np.float64),
            'side_targets': np.asarray([int(row['anatomical_region'] == 'right_hip') for row in prepared['rows']]),
            'side_groups': prepared['groups'], 'side_folds': prepared['side_folds'],
            'side_ids': ids, 'quality_indices': np.flatnonzero(mask), 'quality_ids': ids[mask],
            'side_cache_sha256': sha256(cache), 'side_backbone_sha256': fingerprint['backbone_sha256']}


def feature_weights(task):
    if task not in TASKS:
        raise ValueError('Unknown hip quality task')
    weights = np.ones(FEATURE_COUNTS[task], dtype=np.float64)
    if task == 'roi':
        weights[512:] = np.sqrt(512 / 7)
    return weights


def fit_quality_classifier(task, x, y):
    """Fit train-only scaler/LR; return a safe, already numeric payload."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=int)
    if (x.ndim != 2 or x.shape != (len(y), FEATURE_COUNTS[task]) or not np.isfinite(x).all()
            or set(np.unique(y)) != {0, 1}):
        raise ValueError(f'{task}: valid features and both binary classes are required')
    weights = feature_weights(task)
    with threadpool_limits(limits=2):
        scaler = StandardScaler().fit(x)
        lr = LogisticRegression(**LR_PARAMS, random_state=SEEDS[task]).fit(scaler.transform(x) * weights, y)
    values = {'mean': scaler.mean_.astype(np.float64),
              'scale': (scaler.scale_ / weights).astype(np.float64),
              'coef': lr.coef_.astype(np.float64), 'intercept': lr.intercept_.astype(np.float64),
              'classes': lr.classes_.astype(np.int64)}
    error = float(np.max(np.abs(score_numeric(values, x) - lr.predict_proba(scaler.transform(x) * weights)[:, 1])))
    if error > 1e-12:
        raise AssertionError('Numeric hip logistic payload differs from fitted classifier')
    return values


def score_numeric(values, x):
    x = np.asarray(x, dtype=np.float64)
    logits = ((x - values['mean']) / values['scale']) @ np.asarray(values['coef']).reshape(-1) + float(np.asarray(values['intercept']).reshape(-1)[0])
    probability = np.exp(-np.logaddexp(0., -logits))
    if not np.isfinite(probability).all():
        raise ValueError('Nonfinite hip logistic score')
    return probability


def select_threshold(y, probability, minimum_recall=.70):
    """Max training-crossfit F1; ties BA, precision, largest probability cutoff."""
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if (y.ndim != 1 or probability.shape != y.shape or not len(y)
            or set(np.unique(y)) != {0, 1} or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1)) or not 0 < minimum_recall <= 1):
        raise ValueError('Threshold calibration needs both binary classes and finite probabilities')
    choices = np.unique(np.r_[0., probability, 1.])
    predicted = probability[:, None] >= choices[None, :]
    tp = (predicted & (y[:, None] == 1)).sum(axis=0)
    fp = (predicted & (y[:, None] == 0)).sum(axis=0)
    fn, tn = y.sum() - tp, (y == 0).sum() - fp
    recall = tp / y.sum()
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    ba = .5 * (recall + tn / (y == 0).sum())
    eligible = np.flatnonzero(recall >= minimum_recall)
    threshold = float(choices[max(eligible, key=lambda index: (f1[index], ba[index], precision[index], choices[index]))])
    # With mixed finite logistic probabilities the optimal threshold should be
    # interior; reject degenerate fits instead of publishing a sentinel cutoff.
    if not 0 < threshold < 1:
        raise ValueError('Calibrated hip cutoff is degenerate; inspect training inputs')
    return threshold


def _positioning_fold_features(context, fold, groups, *, model_dir=None):
    """Fit a numeric side LR excluding the entire held global study fold."""
    from .laterality.training import SEED, fit
    from threadpoolctl import threadpool_limits
    side_groups, side_folds = np.asarray(context['side_groups']), np.asarray(context['side_folds'])
    query = side_folds == fold
    excluded = set(side_groups[query])
    side_fit = ~np.isin(side_groups, list(excluded))
    quality_indices = np.asarray(context['quality_indices'], dtype=int)
    if (not np.array_equal(side_groups[quality_indices], groups)
            or not query.any() or not side_fit.any()
            or set(side_groups[side_fit]) & excluded):
        raise ValueError('Side calibration membership does not match quality studies')
    side_x, side_y = np.asarray(context['side_features']), np.asarray(context['side_targets'])
    if (side_x.shape != (len(side_groups), 2048) or not np.isfinite(side_x).all()
            or side_y.shape != side_groups.shape or set(np.unique(side_y[side_fit])) != {0, 1}):
        raise ValueError('Valid frozen features and both classes required for calibration side fit')
    with threadpool_limits(limits=2):
        pipeline = fit(side_x[side_fit].astype(np.float64), side_y[side_fit], .1)
    scaler, lr = pipeline.named_steps['standardscaler'], pipeline.named_steps['logisticregression']
    values = {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': lr.coef_,
              'intercept': lr.intercept_, 'classes': lr.classes_}
    if values['classes'].tolist() != [0, 1]:
        raise ValueError('Unexpected calibration side class binding')
    ids = np.asarray(context['side_ids'])
    proof = {'format': 'hip_cutoff_side_crossfit_v1', 'fold': int(fold),
             'fit_n': int(side_fit.sum()), 'fit_image_ids': ids[side_fit].tolist(),
             'fit_studies': sorted(set(side_groups[side_fit])),
             'held_n': int(query.sum()), 'held_image_ids': ids[query].tolist(),
             'held_studies': sorted(excluded), 'all_held_studies_excluded_from_fit': True,
             'backbone_sha256': context['side_backbone_sha256'],
             'frozen_feature_cache_sha256': context['side_cache_sha256'],
             'LR': {'C': .1, 'class_weight': 'balanced', 'solver': 'liblinear',
                    'max_iter': 2000, 'tol': 1e-6, 'random_state': SEED},
             'probability_threshold': .5}
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        file = model_dir / f'fold{fold}.npz'
        np.savez_compressed(file, **values, metadata_json=np.asarray(json.dumps(proof, sort_keys=True)))
        with np.load(file, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in values}
        proof.update(model_file=f'{model_dir.name}/{file.name}', model_sha256=sha256(file))
    p = score_numeric(values, side_x)
    error = float(np.max(np.abs(p - pipeline.predict_proba(side_x.astype(np.float64))[:, 1])))
    if error > 1e-12:
        raise AssertionError('Numeric calibration side probabilities differ from fitted LR')
    proof['numeric_probability_error'] = error
    normal, mirrored = np.asarray(context['normal']), np.asarray(context['mirrored'])
    if (normal.shape != (len(groups), FEATURE_COUNTS['positioning_rotation'])
            or mirrored.shape != normal.shape or not np.isfinite(normal).all()
            or not np.isfinite(mirrored).all()):
        raise ValueError('Malformed calibration positioning views')
    side_right = p[quality_indices] >= .5
    proof['quality_image_ids'] = np.asarray(context['quality_ids']).tolist()
    proof['quality_predicted_right'] = side_right.astype(int).tolist()
    return np.where(side_right[:, None], mirrored, normal), proof


def crossfit_training_probabilities(task, x, y, folds, groups, *, progress=None,
                                   positioning_context=None, model_dir=None):
    """Five temporary full side/quality fits calibrate the final cutoff.

    Positioning requires independently fitted side preprocessing per held fold.
    These temporary numeric classifiers are never a production ensemble.
    """
    x, y, folds, groups = map(np.asarray, (x, y, folds, groups))
    if set(folds) != set(range(5)):
        raise ValueError('Expected five complete training calibration folds')
    if task == 'positioning_rotation' and positioning_context is None:
        raise ValueError('Positioning calibration requires fold-excluded automatic side preprocessing')
    probability = np.full(len(y), np.nan)
    progress = progress or Progress('hip', False)
    calibration = []
    for fold in range(5):
        train, val = folds != fold, folds == fold
        if set(groups[train]) & set(groups[val]):
            raise ValueError('Study leakage in hip cutoff calibration')
        fold_x, side_proof = x, None
        if task == 'positioning_rotation':
            fold_x, side_proof = _positioning_fold_features(positioning_context, fold, groups, model_dir=model_dir)
            if not set(groups[val]).issubset(set(side_proof['held_studies'])):
                raise ValueError('Side preprocessing fitted a held quality study')
        values = fit_quality_classifier(task, fold_x[train], y[train])
        probability[val] = score_numeric(values, fold_x[val])
        calibration.append({'fold': fold, 'fit_n': int(train.sum()), 'validation_n': int(val.sum()),
                            'fit_studies': sorted(set(groups[train])), 'validation_studies': sorted(set(groups[val])),
                            'automatic_side': side_proof,
                            'calibration_recipe': CALIBRATION_RECIPE})
        progress.update('calibrate', f'{task}: temporary classifier; final inference keeps one LR', fold + 1, 5)
    if not np.isfinite(probability).all():
        raise AssertionError('Incomplete hip crossfit calibration')
    return probability, calibration


def write_classifier(path, values, metadata, threshold):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.fit-') as temporary:
        staged = Path(temporary) / path.name
        np.savez_compressed(staged, **values, threshold=np.asarray(threshold, dtype=np.float64),
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        staged.replace(path)
    return path
