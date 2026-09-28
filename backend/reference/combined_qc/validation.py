"""Cross-fit every learned compact QC classifier without training-set metrics.

The public ImageNet encoders stay frozen. Every learned outer-fold model,
scaler, prevalence and threshold excludes that fold's complete global studies.
Inner hip threshold calibration also cross-fits the automatic side model.
Numeric parameters are saved with pickle disabled and reloaded before scoring.

This is grouped validation of a previously selected recipe, not a sealed test.
The existing coverage/angle algorithms have already been developed on this data
and are reported in a separate, explicitly non-independent table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .common import Progress, quiet_output, write_json
from .hip.preprocessing import acquisition_features, centered_score, letterbox
from .router.features import actual_source, load_record, prepare_native, records, sha256

ROOT = Path(__file__).resolve().parent
TASKS = ("positioning_rotation", "roi")
SEED = 20260928


def _session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.log_severity_level = 3
    for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
        options.add_session_config_entry(key, "0")
    return ort.InferenceSession(str(path), sess_options=options,
                               providers=["CPUExecutionProvider"])


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checked_graph(runtime, filename, manifest, *, nested=False):
    runtime = Path(runtime).resolve()
    spec = _json(runtime / manifest)
    if nested:
        key = "resnet18" if filename.startswith("resnet18") else "spinenet"
        info = spec[key]
        if info["path"] != filename:
            raise ValueError("Unexpected ONNX manifest path")
        expected = info["sha256"]
    else:
        info = spec["files"][filename]
        expected = info["sha256"] if isinstance(info, dict) else info
    path = (runtime / filename).resolve()
    if not path.is_relative_to(runtime) or sha256(path) != expected:
        raise ValueError(f"Frozen ONNX checksum mismatch: {filename}")
    return path


def _metrics(truth, prediction, score=None, *, macro=False):
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y, pred = np.asarray(truth, dtype=int), np.asarray(prediction, dtype=int)
    if not len(y) or pred.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Aligned nonempty binary truth and decisions required")
    if score is not None:
        score = np.asarray(score, dtype=np.float64)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError("Finite aligned continuous scores required")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"n": len(y), "positives": int(y.sum()),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if macro else None,
            "roc_auc": float(roc_auc_score(y, score)) if both and score is not None else None,
            "average_precision": float(average_precision_score(y, score)) if both and score is not None else None,
            "accuracy": float(np.mean(y == pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


LEARNED_SCORE_TYPES = {
    "region": "raw_logistic_probability_spine",
    "side": "raw_logistic_probability_right_hip",
    "hip_positioning_rotation": "fold_threshold_centered_operational_score",
    "hip_roi": "fold_threshold_centered_operational_score",
    "spine_artifacts": "training_prevalence_corrected_probability",
}


def learned_score_diagnostics(predictions):
    """Describe both probability and operational-score OOF ranking.

    A threshold-centred score is monotone within a fold. Different learned
    cutoffs can change rankings across folds, so pooled AUC is not invariant.
    These are diagnostics from the same saved outer predictions, not a new fit.
    """
    values = {key: [] for key in LEARNED_SCORE_TYPES}
    for prediction in predictions:
        fold = int(prediction["outer_fold"])
        probability = prediction["region_probability_spine"]
        values["region"].append((prediction["region_truth"], probability, probability, fold))
        if "side_probability_right" in prediction:
            probability = prediction["side_probability_right"]
            values["side"].append((prediction["side_truth"], probability, probability, fold))
        for short, result in prediction.get("hip", {}).items():
            values["hip_" + short].append((result["truth"], result["probability"], result["score"], fold))
        if "artifacts" in prediction:
            result = prediction["artifacts"]
            values["spine_artifacts"].append((result["truth"], result["probability"], result["probability"], fold))

    from sklearn.metrics import roc_auc_score
    def ranking(items):
        array = np.asarray(items, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
            raise ValueError("Finite aligned OOF probability/score diagnostics required")
        if (not np.isin(array[:, 0], [0, 1]).all()
                or np.any((array[:, 1:3] < 0) | (array[:, 1:3] > 1))
                or not np.isin(array[:, 3], range(5)).all()):
            raise ValueError("Binary truth, probability scores and five valid outer fold IDs required")
        both = len(np.unique(array[:, 0])) == 2
        return {"n": len(array), "positives": int(array[:, 0].sum()),
                "roc_auc_probability": float(roc_auc_score(array[:, 0], array[:, 1])) if both else None,
                "roc_auc_operational_score": float(roc_auc_score(array[:, 0], array[:, 2])) if both else None}

    diagnostics = {}
    for task, items in values.items():
        if not items:
            continue
        diagnostics[task] = {**ranking(items), "operational_score_type": LEARNED_SCORE_TYPES[task],
                "probability_type": "training_prevalence_corrected_probability" if task == "spine_artifacts" else "raw_logistic_probability",
                "pooling": "One excluded-study outer prediction per image; pooled across five fits",
                "per_fold": [{"outer_fold": fold, **ranking([item for item in items if item[3] == fold])}
                             for fold in sorted({item[3] for item in items})]}
    return diagnostics


def annotate_grouped_metrics(grouped, predictions):
    """Attach report semantics without changing decisions or existing metrics."""
    diagnostics = learned_score_diagnostics(predictions)
    bindings = [("region", grouped["region_classifier"], "lumbar_spine", True),
                ("side", grouped["hip_laterality"], "right_hip_visual_orientation", True),
                ("hip_positioning_rotation", grouped["hip"]["positioning_rotation"], "quality_violation", False),
                ("hip_roi", grouped["hip"]["roi"], "quality_violation", False),
                ("spine_artifacts", grouped["spine"]["artifacts"], "quality_violation", False)]
    for task, metrics, positive_class, macro in bindings:
        metrics.update(positive_class=positive_class, displayed_f1_key="macro_f1" if macro else "f1",
                       displayed_f1_averaging="macro" if macro else "binary",
                       precision_recall_averaging="binary", roc_auc_score_type=LEARNED_SCORE_TYPES[task],
                       evaluation_scope="Internal excluded-study OOF of a previously selected training recipe",
                       score_diagnostics=diagnostics[task])
    grouped["hip"]["or"].update(positive_class="any_quality_violation", displayed_f1_key="f1",
                               displayed_f1_averaging="binary", roc_auc_score_type=None,
                               evaluation_scope="Internal excluded-study OOF profile decisions before router rejection")
    return diagnostics


def _make_folds(rows, split):
    """Keep existing five quality folds and extend them to unlabelled studies."""
    by_id = {r["image_id"]: r for r in rows}
    assigned = {}
    if set(split["folds"]) != {str(i) for i in range(5)}:
        raise ValueError("Validation currently requires the saved five-fold protocol")
    for key, value in split["folds"].items():
        for image_id in value["val"]:
            study = by_id[image_id]["study_id"]
            fold = int(key)
            if study in assigned and assigned[study] != fold:
                raise ValueError("A global study crosses saved outer folds")
            assigned[study] = fold
    # No target or model score is used to assign previously excluded studies.
    remaining = sorted({r["study_id"] for r in rows} - assigned.keys())
    counts = np.bincount(list(assigned.values()), minlength=5)
    for study in remaining:
        fold = int(np.argmin(counts))
        assigned[study] = fold
        counts[fold] += 1
    folds = np.array([assigned[r["study_id"]] for r in rows], dtype=int)
    groups = np.array([r["study_id"] for r in rows])
    for fold in range(5):
        if set(groups[folds == fold]) & set(groups[folds != fold]):
            raise AssertionError("Outer global study overlap")
    return folds, groups, {"study_to_fold": assigned, "newly_assigned_studies": remaining,
                            "assignment": "Saved five global study folds; extra studies assigned to smallest fold"}


def _prepare(data, source_data, spine_data, progress):
    rows = records(data)
    split = _json(data / "splits.json")
    if split.get("manifest_sha256") != sha256(data / "manifest.jsonl"):
        raise ValueError("Saved folds do not match the global image manifest")
    folds, groups, fold_proof = _make_folds(rows, split)
    from .spine.train_single import _read_records
    spine_rows, spine_labeled, artifact_truth = _read_records(spine_data)
    spine_targets = {r["image_id"]: r["targets"] for r in
                     [json.loads(line) for line in (spine_data / "labeled_manifest.jsonl").read_text().splitlines() if line.strip()]}
    corrected = {spine_rows[i]["image_id"]: int(y) for i, y in zip(spine_labeled, artifact_truth)}
    global_spine = {r["image_id"] for r in rows if r["anatomical_region"] == "lumbar_spine"}
    if {r["image_id"] for r in spine_rows} != global_spine:
        raise ValueError("Local/global spine images differ; cannot establish correct study grouping")
    native, frames, geometry, trimmed = [], [], [], []
    sources = {}
    for i, row in enumerate(rows):
        pixels = load_record(row, data, source_data)
        frame, geom = letterbox(pixels)
        if any(geom[key] != row[key] for key in geom):
            raise ValueError(f"Native/manifest geometry differs: {row['image_id']}")
        native.append(pixels)
        frames.append(np.rint(frame * 255).astype(np.uint8))
        geometry.append(geom)
        trimmed.append(prepare_native(pixels))
        sources[row["image_id"]] = sha256(actual_source(row, data, source_data))
        progress.update("prepare", "Verified native image pixels", i + 1, len(rows))
    regions = np.array([int(r["anatomical_region"] == "lumbar_spine") for r in rows])
    hip_mask = regions == 0
    side = np.array([int(r["anatomical_region"] == "right_hip") if h else -1
                     for r, h in zip(rows, hip_mask)])
    y = {t: np.array([r["targets"]["hip_" + t] if h and r["targets"]["hip_" + t] is not None else -1
                     for r, h in zip(rows, hip_mask)], dtype=int) for t in TASKS}
    artifacts = np.array([corrected.get(r["image_id"], -1) for r in rows], dtype=int)
    if int(hip_mask.sum()) != 155 or sum(y["roi"] >= 0) != 150 or sum(artifacts >= 0) != 99:
        raise ValueError("Unexpected number of labelled cases")
    if not np.array_equal(y["roi"] >= 0, y["positioning_rotation"] >= 0):
        raise ValueError("Hip subtask label availability differs")
    if int(artifacts[artifacts >= 0].sum()) != 17:
        raise ValueError("Corrected artifact labels must contain 17 positive cases")
    return {"rows": rows, "ids": np.array([r["image_id"] for r in rows]), "groups": groups,
            "folds": folds, "fold_proof": fold_proof, "native": native,
            "frames": np.array(frames), "trimmed": np.array(trimmed), "geometry": geometry,
            "regions": regions, "hip_mask": hip_mask, "side_truth": side, "quality_truth": y,
            "artifact_truth": artifacts, "spine_targets": spine_targets, "source_hashes": sources,
            "patient_independence_verified": bool(split.get("patient_independence_verified", False))}


def _extract(graph, frames, names, progress, label, batch_size=8):
    parts = {name: [] for name in names}
    for start in range(0, len(frames), batch_size):
        arrays = graph.run(names, {"gray_u8": np.ascontiguousarray(frames[start:start + batch_size])})
        for name, array in zip(names, arrays):
            if not np.isfinite(array).all():
                raise ValueError("Non-finite public neural features")
            parts[name].append(array)
        progress.update("features", label, min(start + batch_size, len(frames)), len(frames))
    return {name: np.concatenate(arrays).astype(np.float32) for name, arrays in parts.items()}


def _features(prepared, paths, output, resume, progress):
    fingerprint = {"validation_code_sha256": sha256(Path(__file__)),
                   "preprocessing_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                           [ROOT / "router/features.py", ROOT / "router/preprocessing.py",
                            ROOT / "hip/preprocessing.py", ROOT / "spine/engine/morphology.py"]},
                   "onnx_sha256": {key: sha256(path) for key, path in paths.items()},
                   "source_sha256": prepared["source_hashes"],
                   "image_ids": prepared["ids"].tolist()}
    cache, meta = output / "features.npz", output / "features.json"
    saved_metadata = _json(meta) if meta.is_file() else {}
    if (resume and cache.is_file() and saved_metadata.get("fingerprint") == fingerprint
            and saved_metadata.get("cache_sha256") == sha256(cache)):
        with np.load(cache, allow_pickle=False) as saved:
            result = {name: saved[name].copy() for name in saved.files}
        expected_shapes = {"region": (255, 512), "side": (255, 2048),
                           "hip_normal": (155, 6400), "hip_mirrored": (155, 6400),
                           "hip_roi": (155, 519), "artifact": (100, 2680)}
        if (set(result) == set(expected_shapes) | {"morphology_names_json"}
                and all(result[key].shape == shape for key, shape in expected_shapes.items())
                and all(np.isfinite(a).all() for a in result.values())):
            progress.update("features", "Using verified frozen ONNX feature cache", 1, 1)
            return result, {**fingerprint, "cache_sha256": sha256(cache)}
    # Same trim and mirror averaging as deployed region/side extractors.
    auxiliary = _session(paths["region"])
    original = _extract(auxiliary, prepared["trimmed"], ["global_512", "spatial_2048"], progress, "Auxiliary normal view")
    mirrored = _extract(auxiliary, prepared["trimmed"][:, :, ::-1], ["global_512", "spatial_2048"], progress, "Auxiliary mirrored view")
    result = {"region": ((original["global_512"] + mirrored["global_512"]) / 2).astype(np.float32),
              "side": original["spatial_2048"]}
    # The two auxiliary exports must be the same public graph. If not, use
    # the actual side export rather than assuming numerical identity.
    if sha256(paths["side"]) != sha256(paths["region"]):
        result["side"] = _extract(_session(paths["side"]), prepared["trimmed"], ["spatial_2048"], progress, "Side public graph")["spatial_2048"]
    quality = _session(paths["hip"])
    indices = np.flatnonzero(prepared["hip_mask"])
    frames = prepared["frames"][indices]
    normal = _extract(quality, frames, ["global_512", "stage3_6400"], progress, "Hip native view")
    flipped = _extract(quality, frames[:, :, ::-1], ["stage3_6400"], progress, "Hip mirrored view")
    result["hip_normal"] = normal["stage3_6400"]
    result["hip_mirrored"] = flipped["stage3_6400"]
    result["hip_roi"] = np.c_[normal["global_512"], [acquisition_features(prepared["geometry"][i]) for i in indices]]
    from .spine.engine.morphology import _image_features
    indices = np.flatnonzero(prepared["regions"])
    resnet = _extract(_session(paths["spine_resnet"]), prepared["frames"][indices],
                      ["global_512", "spatial_2048"], progress, "Spine artifact features")
    descriptors = [_image_features(prepared["native"][i]) for i in indices]
    names = list(descriptors[0])
    if len(names) != 120 or any(list(d) != names for d in descriptors):
        raise ValueError("Artifact morphology feature ordering changed")
    result["artifact"] = np.ascontiguousarray(np.c_[resnet["global_512"],
                                      np.array([list(d.values()) for d in descriptors], dtype=np.float32),
                                      resnet["spatial_2048"]], dtype=np.float32)
    # Names are UTF-8 JSON bytes to keep the cache strictly numeric.
    result["morphology_names_json"] = np.frombuffer(json.dumps(names).encode(), dtype=np.uint8)
    np.savez_compressed(cache, **result)
    digest = sha256(cache)
    write_json(meta, {"fingerprint": fingerprint, "cache_sha256": digest})
    return result, {**fingerprint, "cache_sha256": digest}


def _proof(prepared, fit, query, path, *, calibration=None, **extra):
    fit_groups, query_groups = set(prepared["groups"][fit]), set(prepared["groups"][query])
    if fit_groups & query_groups or set(prepared["ids"][fit]) & set(prepared["ids"][query]):
        raise AssertionError("A prediction study participated in fitting")
    value = {"fit_image_ids": prepared["ids"][fit].tolist(),
             "prediction_image_ids": prepared["ids"][query].tolist(),
             "fit_study_ids": sorted(fit_groups), "prediction_study_ids": sorted(query_groups),
             "study_disjoint": True, "classifier_file": str(path.name),
             "classifier_sha256": sha256(path), **extra}
    if calibration is not None:
        calibration_groups = set(prepared["groups"][calibration])
        if calibration_groups & query_groups:
            raise AssertionError("Outer prediction study participated in threshold calibration")
        value["threshold_calibration_image_ids"] = prepared["ids"][calibration].tolist()
        value["threshold_calibration_study_ids"] = sorted(calibration_groups)
    return value


def _fit_auxiliary(x, y, fit, query, file, prepared, *, excluded_groups, role):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y[fit])) != 2:
        raise ValueError(f"{role}: fold training requires both classes")
    if set(prepared["groups"][fit]) & set(excluded_groups):
        raise AssertionError("Auxiliary model fitted a forbidden study")
    scaler = StandardScaler().fit(x[fit].astype(np.float64))
    model = LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                               max_iter=2000, tol=1e-6, random_state=20260927)
    model.fit(scaler.transform(x[fit].astype(np.float64)), y[fit])
    np.savez_compressed(file, mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_[0],
                        intercept=model.intercept_, classes=model.classes_)
    with np.load(file, allow_pickle=False) as saved:
        if saved["classes"].tolist() != [0, 1]:
            raise ValueError("Wrong classifier class binding")
        logits = ((x[query].astype(np.float64) - saved["mean"]) / saved["scale"]) @ saved["coef"] + saved["intercept"][0]
    from scipy.special import expit
    p = expit(logits)
    error = float(np.max(np.abs(p - model.predict_proba(scaler.transform(x[query].astype(np.float64)))[:, 1])))
    if error > 1e-12:
        raise AssertionError("Converted auxiliary NPZ probability mismatch")
    # Queries can include fit images to choose a training representation, so
    # the strict prediction proof is restricted to genuinely excluded groups.
    strict_query = query & np.isin(prepared["groups"], list(excluded_groups))
    return p, _proof(prepared, fit, strict_query, file, role=role,
                     excluded_study_ids=sorted(excluded_groups), numeric_probability_error=error)


def _auxiliary_oof(prepared, features, output, progress):
    n = len(prepared["rows"])
    p_region, p_side = np.full(n, np.nan), np.full(n, np.nan)
    proofs = {"region": [], "laterality": []}
    # Put side features into global image order without inventing spine labels.
    side_x = features["side"].astype(np.float64)
    for outer in range(5):
        test = prepared["folds"] == outer
        excluded = set(prepared["groups"][test])
        p_region[test], proof = _fit_auxiliary(features["region"], prepared["regions"], ~test, test,
                   output / f"region_outer{outer}.npz", prepared, excluded_groups=excluded, role="region")
        proofs["region"].append(dict(outer_fold=outer, **proof))
        fit, query = ~test & prepared["hip_mask"], test & prepared["hip_mask"]
        p_side[query], proof = _fit_auxiliary(side_x, prepared["side_truth"], fit, query,
                   output / f"laterality_outer{outer}.npz", prepared, excluded_groups=excluded, role="laterality")
        proofs["laterality"].append(dict(outer_fold=outer, **proof))
        progress.update("crossfit", "Region and side numeric classifiers", outer + 1, 5)
    if not np.isfinite(p_region).all() or not np.isfinite(p_side[prepared["hip_mask"]]).all():
        raise AssertionError("Missing auxiliary cross-fitted prediction")
    return {"region": p_region, "laterality": p_side}, proofs


def _fit_quality(task, x, y, fit, query, file, prepared, hip_indices, *, encoder_sha256,
                 threshold=.5, threshold_source="Fixed inner prediction cutoff; not used for outer labels"):
    from .hip.features import (FEATURE_COUNTS, FEATURE_VIEWS, LR_PARAMS, SEEDS,
                               fit_quality_classifier, score_numeric, write_classifier)
    from .hip.runtime import PUBLIC_SOURCE_SHA, load_numeric_classifier, score_numeric as runtime_score
    model = fit_quality_classifier(task, x[fit], y[fit])
    # Use exactly the production numeric serializer, then score its parameters.
    metadata = {"format": "hip_compact_numeric_lr_source_v1",
                "purpose": "temporary_study_grouped_cross_validation", "task": task,
                "feature_view": FEATURE_VIEWS[task], "feature_dimension": FEATURE_COUNTS[task],
                "source_weights_sha256": PUBLIC_SOURCE_SHA, "encoder_sha256": encoder_sha256,
                "fit_image_ids": prepared["ids"][hip_indices[fit]].tolist(),
                "fit_study_ids": sorted(set(prepared["groups"][hip_indices[fit]])),
                "prediction_image_ids": prepared["ids"][hip_indices[query]].tolist(),
                "prediction_study_ids": sorted(set(prepared["groups"][hip_indices[query]])),
                "training_n": int(fit.sum()), "positives": int(y[fit].sum()),
                "fit_dtype": "float64", "LR": LR_PARAMS, "seed": SEEDS[task],
                "threshold": float(threshold), "threshold_source": threshold_source}
    write_classifier(file, model, metadata=metadata, threshold=threshold)
    restored, _ = load_numeric_classifier(file, task, expected_metadata=metadata)
    if restored["threshold"] != float(threshold):
        raise AssertionError("Temporary numeric cutoff did not round-trip")
    source = np.asarray(score_numeric(model, x[query]), dtype=np.float64)
    prediction = np.asarray(runtime_score(x[query], restored), dtype=np.float64)
    error = float(np.max(np.abs(prediction - source)))
    if error > 1e-12:
        raise AssertionError("Converted hip numeric classifier mismatch")
    return prediction, error


def _quality_oof(prepared, features, directory, progress, *, encoder_sha256):
    from .hip.features import select_threshold
    hip_indices = np.flatnonzero(prepared["hip_mask"])
    hip_folds = prepared["folds"][hip_indices]
    hip_groups = prepared["groups"][hip_indices]
    n = len(prepared["rows"])
    local_y = {t: prepared["quality_truth"][t][hip_indices] for t in TASKS}
    labelled = local_y["roi"] >= 0
    results = {t: {k: np.full(n, np.nan) for k in ("probability", "score", "threshold", "prediction")} for t in TASKS}
    proofs = {t: [] for t in TASKS}
    side_features = features["side"].astype(np.float64)
    max_error = 0.
    for outer in range(5):
        global_test = prepared["folds"] == outer
        excluded = set(prepared["groups"][global_test])
        global_hip = prepared["hip_mask"]
        # The outer side model excludes all outer test studies. Predicting its
        # own fit images is permitted for quality-training feature selection.
        side_p, side_proof = _fit_auxiliary(side_features, prepared["side_truth"],
                    ~global_test & global_hip, global_hip,
                    directory / f"quality_side_outer{outer}.npz", prepared,
                    excluded_groups=excluded, role="outer_automatic_laterality")
        outer_p = np.where(side_p[:, None] >= .5, features["hip_mirrored"], features["hip_normal"]).astype(np.float64)
        test = (hip_folds == outer) & labelled
        train = (hip_folds != outer) & labelled
        train_global, test_global = np.zeros(n, bool), np.zeros(n, bool)
        train_global[hip_indices[train]], test_global[hip_indices[test]] = True, True
        inner = {t: np.full(len(hip_indices), np.nan) for t in TASKS}
        inner_proofs = {t: [] for t in TASKS}
        for val in range(5):
            if val == outer:
                continue
            forbidden = excluded | set(hip_groups[hip_folds == val])
            global_fit = global_hip & ~np.isin(prepared["groups"], list(forbidden))
            inner_side, side_inner_proof = _fit_auxiliary(side_features, prepared["side_truth"],
                    global_fit, global_hip, directory / f"quality_side_outer{outer}_inner{val}.npz",
                    prepared, excluded_groups=forbidden, role="inner_automatic_laterality")
            position_x = np.where(inner_side[:, None] >= .5, features["hip_mirrored"], features["hip_normal"]).astype(np.float64)
            query = (hip_folds == val) & labelled
            fit = train & ~query
            fit_global, query_global = np.zeros(n, bool), np.zeros(n, bool)
            fit_global[hip_indices[fit]], query_global[hip_indices[query]] = True, True
            for task in TASKS:
                x = position_x if task == "positioning_rotation" else features["hip_roi"]
                file = directory / f"{task}_outer{outer}_inner{val}.npz"
                inner[task][query], error = _fit_quality(task, x, local_y[task], fit, query, file,
                                      prepared, hip_indices, encoder_sha256=encoder_sha256)
                max_error = max(max_error, error)
                inner_proofs[task].append(_proof(prepared, fit_global, query_global, file,
                        inner_fold=val, numeric_probability_error=error,
                        automatic_side=side_inner_proof if task == "positioning_rotation" else None))
        for task in TASKS:
            if not np.isfinite(inner[task][train]).all() or not np.isnan(inner[task][test]).all():
                raise AssertionError("Outer case entered inner threshold calibration")
            threshold = select_threshold(local_y[task][train], inner[task][train],
                                         minimum_recall=.70 if task == "positioning_rotation" else .50)
            x = outer_p if task == "positioning_rotation" else features["hip_roi"]
            file = directory / f"{task}_outer{outer}.npz"
            p, error = _fit_quality(task, x, local_y[task], train, test, file, prepared, hip_indices,
                           encoder_sha256=encoder_sha256, threshold=threshold,
                           threshold_source="Four inner study folds, outer-training studies only")
            max_error = max(max_error, error)
            indices = hip_indices[test]
            result = results[task]
            result["probability"][indices], result["threshold"][indices] = p, threshold
            result["score"][indices] = centered_score(p, threshold)
            result["prediction"][indices] = (p >= threshold).astype(int)
            proofs[task].append(_proof(prepared, train_global, test_global, file,
                    calibration=train_global, outer_fold=outer, threshold=float(threshold),
                    threshold_source="Four inner folds, outer-training studies only",
                    inner_models=inner_proofs[task], numeric_probability_error=error,
                    automatic_side=side_proof if task == "positioning_rotation" else None))
        progress.update("crossfit", "Compact hip; train-only side/scaler/threshold", outer + 1, 5)
    return results, proofs, max_error


def _artifact_oof(prepared, features, directory, progress):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.special import expit, logit
    from .spine.single_logreg import load_numeric_model, save_numeric_model
    from .spine.train_single import PARAMETERS
    spine_indices = np.flatnonzero(prepared["regions"])
    folds = prepared["folds"][spine_indices]
    y = prepared["artifact_truth"][spine_indices]
    labelled = y >= 0
    x = features["artifact"]
    names = json.loads(features["morphology_names_json"].tobytes().decode())
    result = np.full(len(prepared["rows"]), np.nan)
    proofs = []
    for outer in range(5):
        fit, query = (folds != outer) & labelled, (folds == outer) & labelled
        if len(np.unique(y[fit])) != 2:
            raise ValueError("Artifact outer training needs both classes")
        model = Pipeline([("scaler", StandardScaler()),
                          ("classifier", LogisticRegression(**PARAMETERS))])
        model.fit(x[fit], y[fit])
        prevalence = float(y[fit].mean())
        file = directory / f"artifact_outer{outer}.npz"
        save_numeric_model(file, model, train_prevalence=prevalence, morphology_names=names)
        converted = load_numeric_model(file)
        p = converted.predict_proba(x[query])
        source_p = expit(model.decision_function(x[query]) + .5 * logit(prevalence))
        error = float(np.max(np.abs(p - source_p)))
        if error > 1e-6:
            raise AssertionError("Converted artifact classifier probability mismatch")
        result[spine_indices[query]] = p
        fit_global, query_global = np.zeros(len(result), bool), np.zeros(len(result), bool)
        fit_global[spine_indices[fit]], query_global[spine_indices[query]] = True, True
        proofs.append(_proof(prepared, fit_global, query_global, file,
                            outer_fold=outer, threshold=.5, train_prevalence=prevalence,
                            prior_strength=.5, threshold_source="Previously fixed production cutoff",
                            numeric_probability_error=error))
        progress.update("crossfit", "Spine artifact numeric classifier; global study groups", outer + 1, 5)
    if not np.isfinite(result[prepared["artifact_truth"] >= 0]).all():
        raise AssertionError("Missing artifact cross-fitted prediction")
    return result, proofs


def _fixed_rules(prepared, path, progress):
    from .spine.engine.axis import measure_axis
    from .spine.engine.coverage import measure_coverage
    from .spine.landmarks_portable import decode, preprocess
    graph = _session(path)
    values = {}
    indices = np.flatnonzero(prepared["regions"])
    for current, i in enumerate(indices, 1):
        pixels = prepared["native"][i]
        tensor, transform = preprocess(pixels)
        names = ["hm", "reg", "wh"]
        maps = graph.run(names, {"spine_input": np.ascontiguousarray(tensor)})
        detected = decode({name: array[0] for name, array in zip(names, maps)}, transform, threshold=5.)
        values[prepared["ids"][i]] = {"coverage": measure_coverage(pixels), "axis": measure_axis(detected)}
        progress.update("fixed_rules", "Development-only spine algorithms", current, len(indices))
    return values


def _summaries(prepared, aux, quality, artifact, fixed):
    hip_labelled = prepared["quality_truth"]["roi"] >= 0
    artifact_labelled = prepared["artifact_truth"] >= 0
    router_pred = (aux["region"] >= .5).astype(int)
    strict = {
        "region_classifier": _metrics(prepared["regions"], router_pred, aux["region"], macro=True),
        "hip_laterality": _metrics(prepared["side_truth"][prepared["hip_mask"]],
                         aux["laterality"][prepared["hip_mask"]] >= .5,
                         aux["laterality"][prepared["hip_mask"]], macro=True),
        "hip": {task: _metrics(prepared["quality_truth"][task][hip_labelled],
                         quality[task]["prediction"][hip_labelled], quality[task]["score"][hip_labelled]) for task in TASKS},
        "spine": {"artifacts": _metrics(prepared["artifact_truth"][artifact_labelled],
                         artifact[artifact_labelled] >= .5, artifact[artifact_labelled])}}
    strict["hip"]["or"] = _metrics(np.logical_or(*[prepared["quality_truth"][t][hip_labelled] for t in TASKS]),
                         np.logical_or(*[quality[t]["prediction"][hip_labelled] for t in TASKS]))
    region_review = np.maximum(aux["region"], 1 - aux["region"]) < .9
    side_review = np.maximum(aux["laterality"][prepared["hip_mask"]],
                             1 - aux["laterality"][prepared["hip_mask"]]) < .9
    strict["region_classifier"]["needs_review_count"] = int(region_review.sum())
    strict["hip_laterality"]["needs_review_count"] = int(side_review.sum())
    # Branch metrics above include every labelled case. Report routing failures
    # separately; do not silently exclude them or assume a missing QC score=0.
    route_hip_errors = int(np.sum(router_pred[hip_labelled] != 0))
    route_spine_errors = int(np.sum(router_pred[artifact_labelled] != 1))
    hip_unavailable = int(np.sum((router_pred != 0)[hip_labelled] | region_review[hip_labelled]))
    spine_unavailable = int(np.sum((router_pred != 1)[artifact_labelled] | region_review[artifact_labelled]))
    end_to_end = {
        "hip": {"n": int(hip_labelled.sum()), "routing_errors": route_hip_errors,
                "router_needs_review": int(region_review[hip_labelled].sum()),
                "unavailable_cases": hip_unavailable,
                "scored_fraction": float(1 - hip_unavailable / hip_labelled.sum()),
                "metrics": strict["hip"] if not hip_unavailable else None},
        "spine_artifacts": {"n": int(artifact_labelled.sum()), "routing_errors": route_spine_errors,
                "router_needs_review": int(region_review[artifact_labelled].sum()),
                "unavailable_cases": spine_unavailable,
                "scored_fraction": float(1 - spine_unavailable / artifact_labelled.sum()),
                "metrics": strict["spine"]["artifacts"] if not spine_unavailable else None},
        "global_qc_independent_metrics_available": False,
        "reason": "Spine coverage/angle recipes were developed on these same data; OR includes those components"}
    development = {"independent_validation": False,
                   "reason": "Fixed production algorithms were previously selected/calibrated on this dataset",
                   "spine": {}}
    if fixed:
        for task, result_key in (("coverage", "spine_positioning"), ("axis", "spine_axis")):
            truth, pred, score = [], [], []
            missing = 0
            for row in prepared["rows"]:
                target = prepared["spine_targets"].get(row["image_id"], {}).get(result_key)
                if row["anatomical_region"] != "lumbar_spine" or target is None:
                    continue
                value = fixed[row["image_id"]][task]
                if value["violation"] is None:
                    missing += 1
                    continue
                truth.append(target); pred.append(value["violation"]); score.append(value["score"])
            value = _metrics(truth, pred, score) if truth else None
            development["spine"][task] = {"metrics": value, "unavailable_cases": missing,
                                          "independent_validation": False}
    predictions = []
    for i, row in enumerate(prepared["rows"]):
        entry = {"image_id": row["image_id"], "global_study_id": row["study_id"],
                 "outer_fold": int(prepared["folds"][i]), "region_truth": int(prepared["regions"][i]),
                 "region_probability_spine": float(aux["region"][i]), "region_prediction": int(router_pred[i]),
                 "region_needs_review": bool(max(aux["region"][i], 1 - aux["region"][i]) < .9),
                 "source_pixel_hash": row["pixel_hash"], "study_excluded_from_all_prediction_model_fits": True}
        if prepared["hip_mask"][i]:
            entry.update(side_truth=int(prepared["side_truth"][i]),
                         side_probability_right=float(aux["laterality"][i]),
                         automatic_side="right_hip" if aux["laterality"][i] >= .5 else "left_hip",
                         side_needs_review=bool(max(aux["laterality"][i], 1 - aux["laterality"][i]) < .9))
            if hip_labelled[i]:
                entry["hip"] = {t: {"truth": int(prepared["quality_truth"][t][i]),
                         **{key: int(array[i]) if key == "prediction" else float(array[i])
                            for key, array in quality[t].items()}} for t in TASKS}
        elif artifact_labelled[i]:
            entry["artifacts"] = {"truth": int(prepared["artifact_truth"][i]),
                                  "probability": float(artifact[i]), "prediction": int(artifact[i] >= .5), "threshold": .5}
        if row["image_id"] in fixed:
            entry["development_only_fixed_rules"] = fixed[row["image_id"]]
        predictions.append(entry)
    annotate_grouped_metrics(strict, predictions)
    return strict, end_to_end, development, predictions


def _report_markdown(report):
    def table(rows):
        lines = ["| Модель / подзадача | N | F1 | ROC AUC | Accuracy | Balanced accuracy |",
                 "|---|---:|---:|---:|---:|---:|"]
        for label, m, macro in rows:
            value = lambda key: "—" if m.get(key) is None else f"{m[key]:.4f}"
            lines.append(f"| {label} | {m['n']} | {value('macro_f1' if macro else 'f1')} | {value('roc_auc')} | {value('accuracy')} | {value('balanced_accuracy')} |")
        return "\n".join(lines)
    strict = report["grouped_validation"]
    text = ["# Валидация компактного пайплайна", "",
            "```bash", "python -m combined_qc.validate --ci", "```", "",
            "Команда обучает временные модели для проверки и сохраняет их отдельно от рабочих чекпоинтов. "
            "Рабочие финальные модели обучаются на всех доступных метках через `QCPipeline.train_all()`; "
            "они не используются для оценки собственных обучающих снимков.", "",
            "## Служебные модели", "", table([
                ("Таз / позвоночник — macro F1", strict["region_classifier"], True),
                ("Сторона таза — macro F1", strict["hip_laterality"], True)]), "",
            "## Таз — один замороженный ResNet18 и две логрегрессии", "",
            table([(name, strict["hip"][key], False) for name, key in
                   [("Укладка / ротация", "positioning_rotation"), ("ROI", "roi"), ("OR подзадач таза", "or")]]), "",
            "## Позвоночник — обучаемая модель", "",
            table([("Артефакты — одна логрегрессия", strict["spine"]["artifacts"], False)]), "",
            "## Как устроена проверка", "",
            "Пять сохранённых фолдов разделены по глобальным исследованиям. Каждый снимок получает предсказание "
            "от классификаторов, которые не обучались ни на нём, ни на других снимках его исследования. "
            "Стандартизация, балансировка признаков и поправка на долю артефактов используют только обучение. "
            "Пороги таза выбираются по четырём внутренним фолдам внутри обучающей части. "
            "Модель стороны во внутренней калибровке также исключает внешние и внутренние проверочные исследования.", "",
            "Нейронные признаки вычисляет настоящий проверенный ONNX из рабочего комплекта. "
            "Все временные классификаторы экспортируются в NPZ и загружаются обратно до оценки. "
            "`model_proofs.json` содержит списки исследований и снимков для обучения, калибровки и проверки, "
            "контрольные суммы и проверки численной эквивалентности. `predictions.json` содержит все 255 ответов; "
            "при отсутствии разметки снимок остаётся в прогоне, но не входит в соответствующую метрику.", "",
            "Это валидация уже выбранных рецептов. Эти данные ранее использовались в исследованиях параметров, "
            "поэтому результат не является закрытым независимым тестом. Независимость по пациентам не подтверждена. "
            "Сторона таза оценивается по предварительным меткам визуальной ориентации.", "",
            "F1 нарушений — binary для положительного класса «нарушение»; у области и стороны выводится macro F1. "
            "Precision/recall у служебных моделей относятся к позвоночнику и правой визуальной ориентации соответственно. "
            "Строки разных задач с разными долями положительных случаев нельзя ранжировать как превосходство моделей.", "",
            "ROC AUC в основных таблицах использует рабочий score. У таза score сдвинут относительно порога, "
            "выбранного внутри каждого фолда: pooled AUC может отличаться от AUC исходной вероятности. "
            "`score_diagnostics` в JSON содержит оба pooled AUC и AUC каждого внешнего фолда. "
            "У артефактов probability уже включает поправку по распространённости, рассчитанную только на обучении.", "",
            "## Общий инференс", "",
            f"Ошибок маршрутизации среди 150 размеченных тазов: **{report['end_to_end']['hip']['routing_errors']}**; "
            f"среди 99 позвоночников с меткой артефактов: **{report['end_to_end']['spine_artifacts']['routing_errors']}**. "
            f"Отказов маршрутизатора по уверенности ниже 0.9: **{report['end_to_end']['hip']['router_needs_review']}** "
            f"на тазах и **{report['end_to_end']['spine_artifacts']['router_needs_review']}** на позвоночниках. "
            "При неверной маршрутизации или отказе итоговая метрика ветки не рассчитывается с молчаливым исключением снимка. "
            "Валидационного OR всего проекта сейчас нет: в него входят правила позвоночника, разработанные на этом же наборе."]
    development = report["development_only_algorithms"]["spine"]
    if development:
        text += ["", "## Позвоночник — фиксированные алгоритмы, НЕ независимая валидация", "",
                 "Эти значения вынесены отдельно: пороги охвата и рецепт угла ранее выбирались на данном наборе. "
                 "Новые фолды не устраняют этот факт. Для независимой проверки этих алгоритмов нужны новые снимки.", "",
                 table([(name, development[key]["metrics"], False) for name, key in
                        [("Охват — результат на данных разработки", "coverage"), ("Угол — результат на данных разработки", "axis")]
                        if development[key]["metrics"] is not None])]
    text += ["", "Машиночитаемые метрики: `metrics.json`. Сохранённые временные модели: `temporary_models/`. "
             "Для повторного вычисления нейронных признаков используйте `--no-cache`; для пропуска таблицы фиксированных "
             "алгоритмов — `--skip-fixed-rules`. Вывод статусов включается только через `--ci` или `--verbose`.", ""]
    if report.get("complete_tables"):
        text += ["## Все подзадачи и итоговое OR", "",
                 "[Полная таблица](COMPLETE_TABLES.md) объединяет метрики всех подзадач, OR позвоночника "
                 "с валидационными предсказаниями артефактов и OR профилей. Строки с фиксированными правилами "
                 "явно помечены как содержащие данные разработки. Общий запуск отдельно учитывает отказы "
                 "маршрутизатора и покрытие.", ""]
    return "\n".join(text)


def validate(data=None, *, source_data=None, spine_data=None, hip_checkpoints=None,
             spine_checkpoints=None, router_checkpoints=None, output_dir=None,
             resume=True, include_fixed_rules=True, verbose=False, ci=False):
    """Validate five learned models using temporary five-fold numeric exports.

    Production checkpoints are read-only. A caller may select a separate output
    directory, but it must not lie inside any production checkpoint tree.
    """
    with quiet_output(verbose or ci):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            return _validate(data=data, source_data=source_data, spine_data=spine_data,
                   hip_checkpoints=hip_checkpoints, spine_checkpoints=spine_checkpoints,
                   router_checkpoints=router_checkpoints, output_dir=output_dir,
                   resume=resume, include_fixed_rules=include_fixed_rules,
                   verbose=verbose, ci=ci)


def _validate(*, data, source_data, spine_data, hip_checkpoints, spine_checkpoints,
              router_checkpoints, output_dir, resume, include_fixed_rules, verbose, ci):
    started = time.perf_counter()
    data = Path(data).resolve() if data is not None else ROOT / "data/dxa_v1"
    spine_data = Path(spine_data).resolve() if spine_data is not None else ROOT / "spine/data"
    checkpoint_dirs = {
        "hip": Path(hip_checkpoints).resolve() if hip_checkpoints else ROOT / "hip/checkpoints",
        "spine": Path(spine_checkpoints).resolve() if spine_checkpoints else ROOT / "spine/checkpoints",
        "region": Path(router_checkpoints).resolve() if router_checkpoints else ROOT / "router/checkpoints"}
    output = Path(output_dir).resolve() if output_dir is not None else ROOT / "outputs/cross_validation"
    if any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
           for path in checkpoint_dirs.values()):
        raise ValueError("Validation output must be separate from all production checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "temporary_models"
    temporary.mkdir(exist_ok=True)
    progress = Progress("validation", verbose or ci)
    hip_runtime = checkpoint_dirs["hip"] / "runtime"
    paths = {
        "region": _checked_graph(checkpoint_dirs["region"] / "runtime", "resnet18_features.onnx", "manifest.json"),
        "side": _checked_graph(hip_runtime / "laterality", "resnet18_features.onnx", "manifest.json"),
        "hip": _checked_graph(hip_runtime, "resnet18_quality_features.onnx", "manifest.json"),
        "spine_resnet": _checked_graph(checkpoint_dirs["spine"] / "runtime/onnx", "resnet18_features.onnx", "export_manifest.json", nested=True)}
    if include_fixed_rules:
        paths["spinenet"] = _checked_graph(checkpoint_dirs["spine"] / "runtime/onnx", "spinenet_512.onnx", "export_manifest.json", nested=True)
    # Capture every source/runtime model hash before and after this command.
    before = {str(p): sha256(p) for root in checkpoint_dirs.values()
              for p in root.rglob("*") if p.is_file() and p.suffix in {".onnx", ".npz", ".pth", ".pt", ".ckpt"}}
    prepared = _prepare(data, source_data, spine_data, progress)
    features, feature_proof = _features(prepared, paths, output, resume, progress)
    aux, auxiliary_proofs = _auxiliary_oof(prepared, features, temporary, progress)
    quality, quality_proofs, hip_error = _quality_oof(prepared, features, temporary, progress,
                                                   encoder_sha256=sha256(paths["hip"]))
    artifact, artifact_proofs = _artifact_oof(prepared, features, temporary, progress)
    fixed = _fixed_rules(prepared, paths["spinenet"], progress) if include_fixed_rules else {}
    strict, end_to_end, development, predictions = _summaries(prepared, aux, quality, artifact, fixed)
    after = {str(p): sha256(p) for root in checkpoint_dirs.values()
             for p in root.rglob("*") if p.is_file() and p.suffix in {".onnx", ".npz", ".pth", ".pt", ".ckpt"}}
    if before != after:
        raise AssertionError("Production model files changed during validation")
    proofs = {"region": auxiliary_proofs["region"], "laterality": auxiliary_proofs["laterality"],
              "hip": quality_proofs, "spine_artifacts": artifact_proofs}
    report = {
        "format": "compact_qc_grouped_validation_v1", "n_images": len(prepared["rows"]),
        "protocol": {"outer_folds": 5, "group": "global manifest study_id",
                     "final_deployment_classifiers_used_for_scoring": False,
                     "frozen_features": "Verified production ONNX exports of public pretrained weights",
                     "classifier_inference": "Temporary exported numeric NPZ, pickle disabled",
                     "hip_threshold": "Four inner study folds, train-only scaler and automatic side",
                     "outer_test_studies_excluded_from_fit_and_threshold": True,
                     "patient_independence_verified": prepared["patient_independence_verified"],
                     "sealed_independent_test": False,
                     "selection_warning": "Recipes previously selected on this dataset; model-fit exclusion does not undo recipe selection",
                     "laterality_labels": "Provisional visual orientation, not verified patient side",
                     "spine_artifact_labels": "Corrected local spine labeled_manifest; grouping from global DXA manifest",
                     "fold_assignment": prepared["fold_proof"]},
        "grouped_validation": strict, "end_to_end": end_to_end,
        "development_only_algorithms": development,
        "provenance": {"manifest_sha256": sha256(data / "manifest.jsonl"),
                       "splits_sha256": sha256(data / "splits.json"),
                       "corrected_spine_labels_sha256": sha256(spine_data / "labeled_manifest.jsonl"),
                       "features": feature_proof, "production_models_unchanged": True,
                       "production_model_hashes": before, "hip_numeric_probability_error": hip_error},
        "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "model_proofs.json", proofs)
    write_json(output / "predictions.json", predictions)
    if include_fixed_rules:
        from .validation_tables import write_tables
        report["complete_tables"] = write_tables(output, data=data, spine_data=spine_data)
    write_json(output / "metrics.json", report)
    (output / "README.md").write_text(_report_markdown(report), encoding="utf-8")
    progress.update("complete", f"All learned models validated; tables: {output / 'README.md'}", 1, 1)
    return report
