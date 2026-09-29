"""Recreate the router's ONNX + numeric inference package from source weights."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np

from combined_qc.common import Progress, publish_directory, quiet_output, write_json
from .features import DATA, ROOT, RESNET_STATE_SHA, FeatureExtractor, load_record, prepare_native, records, sha256
from .bootstrap import OFFICIAL_SHA, SOURCE_NAME, export_backbone, export_recipe_sha256, inference_threads, load_source_graph


def load_source_classifier(checkpoints_dir):
    """Validate the single safe numeric source fit without training imports."""
    source = Path(checkpoints_dir).resolve() / 'source'
    manifest = json.loads((source / 'classifier_manifest.json').read_text())
    if manifest.get('format') != 'router_source_v1':
        raise ValueError('Unknown router source format')
    if set(manifest.get('files', {})) != {SOURCE_NAME, 'region_logreg.npz'}:
        raise ValueError('Incomplete router source checkpoint inventory')
    for name, expected in manifest['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != expected:
            raise ValueError(f'Router source checksum mismatch: {name}')
    if manifest['files'][SOURCE_NAME] != OFFICIAL_SHA:
        raise ValueError('Source ImageNet weights differ from the pin')
    with np.load(source / 'region_logreg.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'mean', 'scale', 'coef', 'intercept', 'classes', 'metadata'}:
            raise ValueError('Unexpected source classifier fields')
        values = {name: np.asarray(archive[name]).copy() for name in ['mean', 'scale', 'coef', 'intercept', 'classes']}
        spec = json.loads(str(archive['metadata'].item()))
    if spec != manifest['model'] or spec.get('format') != 'image_only_region_logreg_v1':
        raise ValueError('Source classifier metadata mismatch')
    n = 512 if spec.get('representation') == 'global_512' else 2560 if spec.get('representation') == 'global_spatial_2560' else None
    if (n is None or spec.get('n_features') != n or spec.get('class_names') != ['hip', 'lumbar_spine']
            or values['classes'].tolist() != [0, 1]
            or any(values[name].shape != (n,) for name in ['mean', 'scale', 'coef'])
            or values['intercept'].shape != (1,)
            or any(values[name].dtype != np.float64 or not np.isfinite(values[name]).all() for name in ['mean', 'scale', 'coef', 'intercept'])
            or np.any(values['scale'] <= 0)):
        raise ValueError('Invalid source numeric classifier parameters')
    if (spec.get('threshold') != .5 or spec.get('review_threshold') != .9
            or spec.get('preprocessing') != 'trim_exact_black_border_direct320_mirror_mean_v1'
            or spec.get('ImageNet_state_sha256') != RESNET_STATE_SHA
            or not ((spec.get('fit_scope') == 'all_available_labels'
                     and spec.get('trained_on_holdout') is None
                     and spec.get('holdout_image_ids') == []
                     and spec.get('independent_final_checkpoint_validation') is False)
                    or (spec.get('fit_scope') is None and spec.get('trained_on_holdout') is False))
            or set(spec.get('train_image_ids', [])) & set(spec.get('holdout_image_ids', []))):
        raise ValueError('Invalid source classifier recipe or split')
    return values, spec, manifest


def score_numeric(values, spec, features):
    """Stable float64 inference shared with conversion parity checks."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2560 or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,2560] embeddings')
    if spec['representation'] == 'global_512':
        x = x[:, :512]
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        z = ((x - values['mean']) / values['scale']) @ values['coef'] + values['intercept'][0]
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite numeric source classifier result')
    p = np.empty_like(z)
    positive = z >= 0
    p[positive] = 1 / (1 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    p[~positive] = e / (1 + e)
    return p


def _verification_frames(data, count):
    if count:
        data = Path(data).resolve() if data else DATA
        rows = records(data)
        # Deterministic label-balanced sample, used only for export parity.
        ordered = []
        for region in ['lumbar_spine', 'left_hip', 'right_hip']:
            ordered.extend(row for row in rows if row['anatomical_region'] == region)
        selected = []
        for region in ['lumbar_spine', 'left_hip', 'right_hip']:
            row = next((row for row in rows if row['anatomical_region'] == region), None)
            if row and len(selected) < count:
                selected.append(row)
        chosen = {row['image_id'] for row in selected}
        selected.extend(row for row in ordered if row['image_id'] not in chosen)
        selected = selected[:min(count, len(rows))]
        frames = [prepare_native(load_record(row, data)) for row in selected]
        ids = [row['image_id'] for row in selected]
    else:
        frames, ids = [], []
    # Always exercise a symbolic batch >1, also when conversion has no dataset.
    if len(frames) < 2:
        yy, xx = np.indices((320, 320))
        for index in range(2 - len(frames)):
            frames.append(((xx * 3 + yy * 7 + index * 23) % 256).astype(np.uint8))
            ids.append(f'synthetic_dynamic_batch_{index}')
    return np.stack(frames), ids


def convert_all(checkpoints_dir=None, *, data=None, device='auto', verbose=False,
                ci=False, verify_samples=2):
    """Stage, verify and atomically publish exactly one DL + one ML model."""
    if device not in {'auto', 'cpu'}:
        raise ValueError('Router conversion uses CPU')
    if type(verify_samples) is not int or verify_samples < 0:
        raise ValueError('verify_samples must be a nonnegative integer')
    with quiet_output(verbose or ci):
        return _convert_all(checkpoints_dir, data=data, verbose=verbose, ci=ci,
                            verify_samples=verify_samples)


def _convert_all(checkpoints_dir, *, data, verbose, ci, verify_samples):
    import torch
    from .runtime import RegionClassifier

    base = Path(checkpoints_dir).resolve() if checkpoints_dir else ROOT / 'checkpoints'
    values, source_spec, source_manifest = load_source_classifier(base)
    if source_spec.get('backbone_export_recipe_sha256')!=export_recipe_sha256():
        raise ValueError('Backbone export recipe changed after the fit; train again before conversion')
    progress = Progress('router', verbose or ci)
    progress.update('convert', 'Verified one logistic source fit and pinned public backbone')
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix='.conversion-') as temporary:
        staged_base = Path(temporary)
        staged_runtime = staged_base / 'runtime'
        staged_runtime.mkdir()
        export = export_backbone(base, staged_runtime / 'resnet18_features.onnx', verbose=verbose, ci=ci)
        spec = {**source_spec, 'backbone_sha256': export['sha256']}
        numeric_path = staged_runtime / 'region_logreg.npz'
        np.savez_compressed(numeric_path, **values, metadata=np.asarray(json.dumps(spec)))
        write_json(staged_runtime / 'manifest.json', {
            'format': 'region_classifier_runtime_v1',
            'files': {'resnet18_features.onnx': export['sha256'], 'region_logreg.npz': sha256(numeric_path)},
            'model': spec,
        })
        restored = RegionClassifier(staged_base, verbose=verbose, ci=ci)
        frames, ids = _verification_frames(data, verify_samples)
        graph, _ = load_source_graph(base)
        augmented = np.concatenate([frames, frames[:, :, ::-1]])
        with inference_threads(), torch.inference_mode():
            normal_global, normal_spatial = graph(torch.from_numpy(np.ascontiguousarray(augmented)))
            joined = torch.cat([normal_global, normal_spatial], dim=1).cpu().numpy()
        n = len(frames)
        reference_features = ((joined[:n] + joined[n:]) / 2).astype(np.float32)
        converted_features = restored.extractor.extract(frames)
        max_feature_error = float(np.max(np.abs(reference_features - converted_features)))
        if not np.allclose(reference_features, converted_features, rtol=3e-4, atol=3e-4):
            raise ValueError('Router ONNX features differ from source PyTorch')
        reference_probability = score_numeric(values, source_spec, reference_features)
        converted_probability = restored.score_features(converted_features)
        numeric_only = restored.score_features(reference_features)
        numeric_error = float(np.max(np.abs(numeric_only - reference_probability)))
        end_to_end_error = float(np.max(np.abs(converted_probability - reference_probability)))
        if numeric_error > 1e-12 or end_to_end_error > 3e-5:
            raise ValueError('Router numeric/exported probability parity failed')
        if not np.array_equal(converted_probability >= .5, reference_probability >= .5):
            raise ValueError('Router export changed anatomy decisions')
        report = {
            'status': 'converted', 'checkpoints_dir': str(base), 'runtime_dir': str(base / 'runtime'),
            'models': {'resnet18_features': export,
                       'region_logreg': {'source_sha256': source_manifest['files']['region_logreg.npz'],
                                         'runtime_sha256': sha256(numeric_path), 'format': 'numeric_float64_npz',
                                         'count': 1, 'refitted_during_conversion': False}},
            'parity': {'verified_images': ids, 'batch_size_tested': len(augmented),
                       'max_feature_absolute_difference': max_feature_error,
                       'numeric_only_max_probability_difference': numeric_error,
                       'end_to_end_max_probability_difference': end_to_end_error,
                       'all_decisions_match': True},
            'source_inventory': source_manifest['files'], 'opset': export['opset'],
            'regenerated_onnx_from_source': True,
        }
        write_json(staged_runtime / 'conversion_report.json', report)
        progress.update('verify', 'PyTorch/ONNX features and numeric probabilities agree')
        publish_directory(staged_runtime, base / 'runtime')
    progress.update('convert', 'Published validated ONNX + one numeric logistic model', 2, 2)
    return report
