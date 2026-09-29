"""Convert source-only side weights into a self-contained hip bundle child."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np

from combined_qc.common import Progress, publish_directory, quiet_output, write_json
from .features import DATA, DEFAULT_SOURCE, PREPROCESSING, REPRESENTATIONS, RESNET_STATE_SHA, FeatureExtractor, hip_records, load_record, prepare_native, sha256, subset
from .bootstrap import OFFICIAL_SHA, SOURCE_NAME, export_backbone, export_recipe_sha256, inference_threads, load_source_graph


def load_source_classifier(source_dir=None):
    """Validate one source fit using safe numeric arrays, never pickle."""
    source = Path(source_dir).resolve() if source_dir else DEFAULT_SOURCE
    manifest = json.loads((source / 'classifier_manifest.json').read_text())
    if (manifest.get('format') != 'hip_laterality_source_v1'
            or set(manifest.get('files', {})) != {SOURCE_NAME, 'laterality_logreg.npz'}):
        raise ValueError('Invalid hip laterality source inventory')
    for name, expected in manifest['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != expected:
            raise ValueError(f'Laterality source checksum mismatch: {name}')
    if manifest['files'][SOURCE_NAME] != OFFICIAL_SHA:
        raise ValueError('Laterality public ImageNet source differs from the pin')
    with np.load(source / 'laterality_logreg.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'mean', 'scale', 'coef', 'intercept', 'classes', 'metadata'}:
            raise ValueError('Invalid laterality source numeric fields')
        values = {name: np.asarray(saved[name]).copy() for name in ['mean', 'scale', 'coef', 'intercept', 'classes']}
        spec = json.loads(str(saved['metadata'].item()))
    n = REPRESENTATIONS.get(spec.get('representation'))
    if (spec != manifest.get('model') or spec.get('format') != 'image_only_laterality_logreg_v1'
            or spec.get('class_names') != ['left_hip', 'right_hip'] or values['classes'].tolist() != [0, 1]
            or n is None or spec.get('n_features') != n
            or any(values[name].shape != (n,) for name in ['mean', 'scale', 'coef'])
            or values['intercept'].shape != (1,)
            or any(values[name].dtype != np.float64 or not np.isfinite(values[name]).all() for name in ['mean', 'scale', 'coef', 'intercept'])
            or np.any(values['scale'] <= 0)):
        raise ValueError('Invalid laterality source numeric model or metadata')
    if (spec.get('preprocessing') != PREPROCESSING or spec.get('threshold') != .5
            or spec.get('review_threshold') != .9 or spec.get('mirror_average') is not False
            or spec.get('ImageNet_state_sha256') != RESNET_STATE_SHA
            or not ((spec.get('fit_scope') == 'all_available_labels'
                     and spec.get('trained_on_holdout') is None
                     and spec.get('holdout_image_ids') == []
                     and spec.get('independent_final_checkpoint_validation') is False)
                    or (spec.get('fit_scope') is None and spec.get('trained_on_holdout') is False))
            or set(spec.get('train_image_ids', [])) & set(spec.get('holdout_image_ids', []))):
        raise ValueError('Invalid laterality source recipe or split')
    return values, spec, manifest


def score_numeric(values, spec, features):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 2560 or not len(features) or not np.isfinite(features).all():
        raise ValueError('Expected finite [N,2560] ResNet18 embeddings')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            z = ((subset(features, spec['representation']) - values['mean']) / values['scale']) @ values['coef'] + values['intercept'][0]
    except FloatingPointError as error:
        raise ValueError('Laterality numeric normalization overflow') from error
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite laterality log odds')
    probability = np.empty_like(z)
    positive = z >= 0
    probability[positive] = 1 / (1 + np.exp(-z[positive]))
    exp = np.exp(z[~positive])
    probability[~positive] = exp / (1 + exp)
    return probability


def _verification_frames(data, count):
    frames, ids = [], []
    if count:
        data = Path(data).resolve() if data else DATA
        rows = hip_records(data)
        selected = []
        for side in ['left_hip', 'right_hip']:
            row = next((row for row in rows if row['anatomical_region'] == side), None)
            if row and len(selected) < count:
                selected.append(row)
        selected_ids = {row['image_id'] for row in selected}
        selected.extend(row for row in rows if row['image_id'] not in selected_ids)
        selected = selected[:min(count, len(rows))]
        frames = [prepare_native(load_record(row, data)) for row in selected]
        ids = [row['image_id'] for row in selected]
    if len(frames) < 2:
        yy, xx = np.indices((320, 320))
        for index in range(2 - len(frames)):
            frames.append(((xx * 3 + yy * 7 + index * 23) % 256).astype(np.uint8))
            ids.append(f'synthetic_dynamic_batch_{index}')
    return np.stack(frames), ids


def export_runtime(source_dir, target_runtime_dir, *, data=None, verbose=False,
                   ci=False, verify_samples=2):
    """Regenerate and validate this child bundle; never refit the classifier.

    The parent hip converter normally passes a directory inside its own staging
    bundle. Publishing that entire hip bundle remains the parent's operation.
    """
    if type(verify_samples) is not int or verify_samples < 0:
        raise ValueError('verify_samples must be a nonnegative integer')
    with quiet_output(verbose or ci):
        return _export_runtime(source_dir, target_runtime_dir, data=data,
                               verbose=verbose, ci=ci, verify_samples=verify_samples)


def _export_runtime(source_dir, target_runtime_dir, *, data, verbose, ci, verify_samples):
    import torch
    from .runtime import LateralityClassifier

    source = Path(source_dir).resolve()
    target = Path(target_runtime_dir).resolve()
    values, source_spec, source_manifest = load_source_classifier(source)
    if source_spec.get('backbone_export_recipe_sha256') != export_recipe_sha256():
        raise ValueError('Laterality backbone recipe changed after the fit; train again before conversion')
    progress = Progress('hip_laterality', verbose or ci)
    progress.update('convert', 'Verified one numeric source fit and public ImageNet weights')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.laterality-conversion-') as temporary:
        staged = Path(temporary) / 'laterality'
        staged.mkdir()
        export = export_backbone(source, staged / 'resnet18_features.onnx', verbose=verbose, ci=ci)
        spec = {**source_spec, 'backbone_sha256': export['sha256']}
        numeric_path = staged / 'laterality_logreg.npz'
        np.savez_compressed(numeric_path, **values, metadata=np.asarray(json.dumps(spec)))
        write_json(staged / 'manifest.json', {'format': 'hip_laterality_runtime_v1',
                   'files': {'resnet18_features.onnx': export['sha256'], 'laterality_logreg.npz': sha256(numeric_path)},
                   'model': spec})
        restored = LateralityClassifier(runtime_dir=staged, verbose=verbose, ci=ci)
        frames, ids = _verification_frames(data, verify_samples)
        graph, _ = load_source_graph(source)
        with inference_threads(), torch.inference_mode():
            global_features, spatial_features = graph(torch.from_numpy(np.ascontiguousarray(frames)))
            reference_features = torch.cat([global_features, spatial_features], dim=1).cpu().numpy().astype(np.float32)
        converted_features = restored.extractor.extract(frames)
        feature_error = float(np.max(np.abs(reference_features - converted_features)))
        if not np.allclose(reference_features, converted_features, rtol=3e-4, atol=3e-4):
            raise ValueError('Laterality ONNX features differ from source PyTorch')
        reference_probability = score_numeric(values, source_spec, reference_features)
        numeric_only = restored.score_features(reference_features)
        converted_probability = restored.score_features(converted_features)
        numeric_error = float(np.max(np.abs(reference_probability - numeric_only)))
        end_error = float(np.max(np.abs(reference_probability - converted_probability)))
        if numeric_error > 1e-12 or end_error > 3e-5:
            raise ValueError('Laterality numeric/exported probability parity failed')
        if not np.array_equal(reference_probability >= .5, converted_probability >= .5):
            raise ValueError('Laterality conversion changed predicted side')
        report = {'status': 'converted', 'source_dir': str(source), 'runtime_dir': str(target),
                  'regenerated_onnx_from_source': True, 'refitted_during_conversion': False,
                  'models': {'resnet18_features': export,
                             'laterality_logreg': {'source_sha256': source_manifest['files']['laterality_logreg.npz'],
                                                   'runtime_sha256': sha256(numeric_path), 'format': 'numeric_float64_npz', 'count': 1}},
                  'parity': {'verified_images': ids, 'batch_size_tested': len(frames),
                             'mirror_average': False, 'max_feature_absolute_difference': feature_error,
                             'numeric_only_max_probability_difference': numeric_error,
                             'end_to_end_max_probability_difference': end_error, 'all_decisions_match': True},
                  'label_interpretation': 'Agreement with provisional dataset visual side labels; independently verified patient laterality is unavailable'}
        write_json(staged / 'conversion_report.json', report)
        progress.update('verify', 'ONNX and source probabilities agree; side labels preserved')
        publish_directory(staged, target)
    progress.update('convert', 'Validated laterality child bundle ready', 2, 2)
    return report
