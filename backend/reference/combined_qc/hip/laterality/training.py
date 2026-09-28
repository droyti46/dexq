"""Fit one fixed hip-side LR on all155 labels; grouped validation is separate."""
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from combined_qc.common import Progress, quiet_output, write_json
from .bootstrap import SOURCE_NAME, export_recipe_sha256, prepare_backbone
from .features import DATA, DEFAULT_SOURCE, ROOT, PREPROCESSING, REPRESENTATIONS, RESNET_STATE_SHA, FeatureExtractor, actual_source, hip_records, load_record, prepare_native, recipe_fingerprint, sha256, subset

SEED = 20260927
C_VALUES = (.1, 1., 10.)


def fit(x, y, c):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000, tol=1e-6, random_state=SEED))
    return model.fit(x, y)


def metrics(y, probability):
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != y.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid right-hip scores")
    pred = (probability >= .5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    keep = np.maximum(probability, 1 - probability) >= .9
    return {"n_images": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc_right": float(roc_auc_score(y, probability)),
            "confusion_matrix_left_right": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "n": int(support[i])} for i, name in enumerate(("left_hip", "right_hip"))},
            "minimum_true_label_score": float(np.where(y == 1, probability, 1 - probability).min()),
            "review_threshold": .9, "review_count": int((~keep).sum()),
            "accepted_coverage": float(keep.mean()),
            "accepted_accuracy": float(accuracy_score(y[keep], pred[keep])) if keep.any() else None}


def choose_c(x, y, groups, representation):
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED + 1).split(x, y, groups))
    scores = []
    for c in C_VALUES:
        probability = np.full(len(y), np.nan)
        for train, val in folds:
            if set(groups[train]) & set(groups[val]):
                raise ValueError("Study leakage in inner folds")
            model = fit(subset(x[train], representation), y[train], c)
            probability[val] = model.predict_proba(subset(x[val], representation))[:, 1]
        scores.append({"C": c, "metrics": metrics(y, probability)})
    best = max(scores, key=lambda item: (item["metrics"]["macro_f1"], item["metrics"]["balanced_accuracy"], item["metrics"]["roc_auc_right"], -item["C"]))
    return best["C"], scores


def _feature_cache(rows, data, backbone, base, progress):
    cache = base / "features.npz"
    from combined_qc.router import features as shared_features
    fingerprint = {"manifest_sha256": sha256(data / "manifest.jsonl"), "backbone_sha256": sha256(backbone),
                   "feature_code_sha256": sha256(ROOT / "features.py"), "preprocessing": PREPROCESSING,
                   "shared_preprocessing_code_sha256": sha256(Path(shared_features.__file__)),
                   "source_files_sha256": {row["image_id"]: sha256(actual_source(row, data)) for row in rows}}
    metadata = cache.with_suffix(".json")
    if cache.is_file() and metadata.is_file():
        saved_metadata = json.loads(metadata.read_text())
        if saved_metadata.get("fingerprint") == fingerprint and saved_metadata.get("cache_sha256") == sha256(cache):
            with np.load(cache, allow_pickle=False) as saved:
                features = saved["features"]
                if saved["image_ids"].tolist() == [row["image_id"] for row in rows] and features.shape == (155, 2560) and np.isfinite(features).all():
                    progress.update("features", "Using verified non-mirrored image embeddings", 155, 155)
                    return features, fingerprint
    extractor = FeatureExtractor(backbone,expected_sha=sha256(backbone))
    parts = []
    for start in range(0, len(rows), 16):
        frames = np.stack([prepare_native(load_record(row, data)) for row in rows[start:start + 16]])
        parts.append(extractor.extract(frames))
        progress.update("features", "Frozen ResNet18; horizontal orientation retained", min(start + 16, len(rows)), len(rows))
    features = np.concatenate(parts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, image_ids=np.asarray([row["image_id"] for row in rows]))
    write_json(metadata, {"fingerprint": fingerprint, "cache_sha256": sha256(cache)})
    return features, fingerprint


def train(data=None, *, source_dir=None, checkpoints_dir=None, verbose=False,
          ci=False, resume=True, convert=False, target_runtime_dir=None):
    """Fit the hip-side source component; default never changes live runtime.

    source_dir is a direct folder, usually HIPBASE/source/laterality.
    checkpoints_dir is an alias for that same direct folder, not a HIP root.
    The parent hip converter exports/publishes the complete inference bundle.
    """
    if source_dir is not None and checkpoints_dir is not None and Path(source_dir).resolve()!=Path(checkpoints_dir).resolve():
        raise ValueError('source_dir and checkpoints_dir refer to different laterality sources')
    base=Path(source_dir or checkpoints_dir).resolve() if source_dir is not None or checkpoints_dir is not None else DEFAULT_SOURCE
    if convert and target_runtime_dir is None:
        raise ValueError('Pass target_runtime_dir for child conversion, or use the complete hip converter')
    with quiet_output(verbose or ci):
        return _train(data,base=base,verbose=verbose,ci=ci,resume=resume,
                      convert=convert,target_runtime_dir=target_runtime_dir)


def _train(data, *, base, verbose, ci, resume, convert, target_runtime_dir):
    data = Path(data).resolve() if data else DATA
    rows = hip_records(data)
    split = json.loads((ROOT / "split.json").read_text())
    if sha256(data / "manifest.jsonl") != split["manifest_sha256"]:
        raise ValueError("Data changed after the side split was frozen")
    recipe = recipe_fingerprint()
    report_path = base / "training_report.json"
    progress = Progress("hip_laterality", verbose or ci)
    if resume and report_path.is_file():
        from .conversion import load_source_classifier
        _,spec,_=load_source_classifier(base)
        if spec["training_recipe_sha256"] != recipe or spec["split_sha256"] != sha256(ROOT / "split.json"):
            raise ValueError("Laterality recipe changed; use resume=False")
        current = {row["image_id"]: sha256(actual_source(row, data)) for row in rows}
        if current != spec["source_files_sha256"]:
            raise ValueError("Native image inputs changed; use resume=False")
        metadata_path=base/'features.json'
        if (not metadata_path.is_file() or sha256(metadata_path)!=spec.get('feature_cache_metadata_sha256')
                or sha256(base/'features.npz')!=spec.get('feature_cache_sha256')
                or sha256(base/'features_backbone.onnx')!=spec['backbone_sha256']):
            raise ValueError('Laterality source features changed; use resume=False')
        fingerprint=json.loads(metadata_path.read_text())['fingerprint']
        if (fingerprint.get('source_files_sha256')!=current
                or fingerprint.get('feature_code_sha256')!=sha256(ROOT/'features.py')
                or fingerprint.get('manifest_sha256')!=spec['manifest_sha256']
                or fingerprint.get('backbone_sha256')!=spec['backbone_sha256']):
            raise ValueError('Laterality source feature metadata changed; use resume=False')
        progress.update("train", "Reusing verified single hip-side source fit")
        report=json.loads(report_path.read_text())
        report['converted']=False
        if convert:
            from .conversion import export_runtime
            report['conversion']=export_runtime(base,target_runtime_dir,data=data,verbose=verbose,ci=ci)
            report['converted']=True
        write_json(report_path,report)
        return report
    started = time.perf_counter()
    ids = [row['image_id'] for row in rows]
    y = np.asarray([int(row['anatomical_region'] == 'right_hip') for row in rows])
    backbone = prepare_backbone(base, verbose=verbose, ci=ci)
    x, fingerprint = _feature_cache(rows, data, backbone, base, progress)
    representation, c = 'spatial_2048', .1
    progress.update('train', 'One final side LR: all155 labels, spatial2048, C=.1', 0, 1)
    model = fit(subset(x, representation), y, c)
    scaler, lr = model.named_steps['standardscaler'], model.named_steps['logisticregression']
    spec = {'format':'image_only_laterality_logreg_v1', 'class_names':['left_hip','right_hip'],
        'representation':representation, 'n_features':2048, 'C':c, 'seed':SEED,
        'threshold':.5, 'review_threshold':.9, 'mirror_average':False, 'preprocessing':PREPROCESSING,
        'backbone_sha256':sha256(backbone), 'ImageNet_state_sha256':RESNET_STATE_SHA,
        'training_recipe_sha256':recipe, 'backbone_export_recipe_sha256':export_recipe_sha256(),
        'feature_cache_metadata_sha256':sha256(base/'features.json'),
        'feature_cache_sha256':sha256(base/'features.npz'),
        'manifest_sha256':sha256(data/'manifest.jsonl'), 'split_sha256':sha256(ROOT/'split.json'),
        'train_image_ids':ids, 'holdout_image_ids':[], 'trained_on_holdout':None,
        'fit_scope':'all_available_labels', 'independent_final_checkpoint_validation':False,
        'historical_holdout_included_in_final_fit':True,
        'validation_command':'python -m combined_qc.validate',
        'source_files_sha256':fingerprint['source_files_sha256'],
        'label_status':'153 provisional_visual and2 filename_reference; not verified patient laterality'}
    source_model = base/'laterality_logreg.npz'
    with tempfile.TemporaryDirectory(dir=base, prefix='.fit-') as temporary:
        staged = Path(temporary)/source_model.name
        np.savez_compressed(staged, mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64),
            coef=lr.coef_[0].astype(np.float64), intercept=lr.intercept_.astype(np.float64),
            classes=lr.classes_, metadata=np.asarray(json.dumps(spec)))
        staged.replace(source_model)
    write_json(base/'classifier_manifest.json', {'format':'hip_laterality_source_v1', 'files':{
        SOURCE_NAME:sha256(base/SOURCE_NAME), 'laterality_logreg.npz':sha256(source_model)}, 'model':spec})
    from .conversion import load_source_classifier, score_numeric
    values, restored, _ = load_source_classifier(base)
    probability = score_numeric(values, restored, x)
    error = float(np.max(np.abs(probability-model.predict_proba(subset(x, representation))[:,1])))
    if error > 1e-10:
        raise AssertionError('Numeric side model differs from its source LR')
    report = {'protocol':'Fixed recipe; final fit on all155 side labels; grouped validation runs separately',
        'label_interpretation':'Agreement with provisional visual side labels, not independently verified patient anatomy',
        'fit_image_ids':ids, 'held_out_image_ids':[], 'fit_scope':'all_available_labels',
        'selected':{'representation':representation, 'C':c, 'parameter_search':False},
        'holdout':None, 'independent_final_checkpoint_validation':False,
        'validation_command':spec['validation_command'], 'numeric_conversion_max_error':error,
        'training_recipe_sha256':recipe, 'mirror_average':False,
        'seconds':time.perf_counter()-started, 'source_dir':str(base), 'converted':False,
        'limitations':['Final all-data weights have no independent local holdout',
                       'Use temporary excluded-fold models for validation', 'Physical patient side is unverified']}
    write_json(report_path, report)
    write_json(base/'outputs/metrics.json', report)
    progress.update('train', 'All-data side classifier saved; validation scores are separate', 1, 1)
    if convert:
        from .conversion import export_runtime
        report['conversion'] = export_runtime(base, target_runtime_dir, data=data, verbose=verbose, ci=ci)
        report['converted'] = True
        write_json(report_path, report)
        write_json(base/'outputs/metrics.json', report)
    return report
