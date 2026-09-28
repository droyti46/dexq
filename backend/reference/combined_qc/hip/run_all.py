"""ONNX-only inference for all 155 hips and held-out fold metrics on 150 labels.

Full predictions use the five-fold deployment aggregate. Evaluation selects
only the saved outer fold's raw probability and threshold from those results;
its metrics are not metrics for the aggregate. ROI calibration reused some
validation labels and is therefore not a strictly nested unbiased evaluation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import Progress, image_payload, write_json
from .annotation import annotate_image
from .pipeline import DATA_ROOT, HIP_TASKS, PROJECT_ROOT, TASK_NAMES, _geometry, infer, load_checkpoints
from .preprocessing import letterbox, load_native, validate_geometry

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "all"


def _read_cohort(data_root):
    manifest = data_root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted((row for row in rows if row.get("anatomical_region") in {"left_hip", "right_hip"}),
                  key=lambda row: row["image_id"])
    if len(rows) != 155 or len({row["image_id"] for row in rows}) != 155:
        raise ValueError(f"Expected all 155 unique hip images, found {len(rows)}")
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if splits.get("manifest_sha256") != manifest_hash:
        raise ValueError("Hip splits.json does not match manifest.jsonl")
    specs = splits.get("folds")
    if isinstance(specs, dict):
        fold_items = specs.items()
    elif isinstance(specs, list):
        fold_items = ((item.get("fold", index), item) for index, item in enumerate(specs))
    else:
        raise ValueError("Hip splits must define five outer folds")
    assignments = {}
    for key, spec in fold_items:
        fold = int(key)
        if not 0 <= fold < 5:
            raise ValueError(f"Invalid outer fold: {fold}")
        for image_id in spec.get("val", spec.get("val_image_ids", [])):
            if image_id in assignments:
                raise ValueError(f"Duplicate outer-fold assignment: {image_id}")
            assignments[image_id] = fold
    labeled = {row["image_id"]: row for row in rows
               if row.get("eligible") and row.get("source_split") == "train"}
    if len(labeled) != 150:
        raise ValueError(f"Expected the 150 eligible labeled hips, found {len(labeled)}")
    for row in rows:
        validate_geometry(row)
        if not (data_root / row["image_path"]).is_file():
            raise FileNotFoundError(data_root / row["image_path"])
    for image_id, row in labeled.items():
        if image_id not in assignments or any(row["targets"].get(task) not in (0, 1) for task in HIP_TASKS):
            raise ValueError(f"Missing hip label or held-out fold: {image_id}")
    if {assignments[image_id] for image_id in labeled} != set(range(5)):
        raise ValueError("All five outer folds must be represented")
    for fold in range(5):
        train_studies = {row["study_id"] for image_id, row in labeled.items() if assignments[image_id] != fold}
        test_studies = {row["study_id"] for image_id, row in labeled.items() if assignments[image_id] == fold}
        if train_studies & test_studies:
            raise ValueError(f"Study leakage in hip outer fold {fold}")
    return rows, labeled, assignments


def _metric_bundle(truth, predictions, scores):
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(truth) != 150 or predictions.shape != truth.shape or scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError("Expected 150 finite held-out hip predictions")
    tp = int(np.sum((truth == 1) & (predictions == 1)))
    fp = int(np.sum((truth == 0) & (predictions == 1)))
    fn = int(np.sum((truth == 1) & (predictions == 0)))
    tn = int(np.sum((truth == 0) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    positive, negative = scores[truth == 1], scores[truth == 0]
    auc = float(np.mean((positive[:, None] > negative).astype(float) + .5 * (positive[:, None] == negative))) if len(positive) and len(negative) else None
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"n": len(truth), "positives": int(truth.sum()),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "roc_auc": auc,
            "balanced_accuracy": float(np.mean([value for value in (sensitivity, specificity) if value is not None])),
            "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _evaluate_full(labeled, assignments, full_results):
    """Select the original raw held-out fold score, never the aggregate score."""
    by_id = {result["metadata"]["image_id"]: result for result in full_results}
    if len(by_id) != 155 or not set(labeled).issubset(by_id):
        raise ValueError("Full hip inference must contain 155 unique dataset IDs")
    oof = []
    for image_id in sorted(labeled):
        row = labeled[image_id]
        fold = assignments[image_id]
        result = {"image_id": image_id, "study_id": row["study_id"], "fold": fold,
                  "laterality": by_id[image_id]["laterality"],
                  "dataset_laterality": row["anatomical_region"]}
        for task, short in TASK_NAMES.items():
            matching = [item for item in by_id[image_id]["tasks"][task]["folds"] if item["fold"] == fold]
            if len(matching) != 1:
                raise ValueError(f"Missing held-out ONNX fold: {image_id}/{task}")
            probability, threshold = matching[0]["probability"], matching[0]["threshold"]
            if not np.isfinite(probability) or not np.isfinite(threshold) or not 0 <= probability <= 1 or not 0 < threshold < 1:
                raise ValueError(f"Invalid held-out probability: {image_id}/{task}")
            result.update({f"{short}_true": int(row["targets"][task]),
                           f"{short}_probability": float(probability),
                           f"{short}_threshold": float(threshold),
                           f"{short}_prediction": int(probability >= threshold)})
        oof.append(result)
    metrics = {task: _metric_bundle([row[f"{short}_true"] for row in oof],
                                  [row[f"{short}_prediction"] for row in oof],
                                  [row[f"{short}_probability"] for row in oof])
               for task, short in TASK_NAMES.items()}
    return oof, metrics


def _custom_prepared_result(data_root, row, model):
    """Preserve original geometry when prepared files live outside default data."""
    path = (data_root / row["image_path"]).resolve()
    pixels = load_native(path)
    if pixels.shape != (320, 320):
        raise ValueError(f"Prepared hip PNG must be 320x320: {path}")
    prepared = pixels.astype(np.float32) / 255.
    geometry = validate_geometry(row)
    native_path = data_root / row.get("native_path", "__missing__")
    if not native_path.is_file():
        native_path = PROJECT_ROOT / "source_data" / row.get("canonical_source_path", "__missing__")
    if native_path.is_file():
        native = load_native(native_path)
        expected, expected_geometry = letterbox(native)
        if geometry != expected_geometry or not np.array_equal(expected, prepared):
            raise ValueError(f"Prepared image no longer matches its native export: {path}")
        visualization_source = "native_dataset_export"
    else:
        content = pixels[geometry["pad_top"]:geometry["pad_top"] + geometry["resized_height"],
                         geometry["pad_left"]:geometry["pad_left"] + geometry["resized_width"]]
        native = np.asarray(Image.fromarray(content).resize((geometry["original_width"], geometry["original_height"]), Image.Resampling.BILINEAR))
        visualization_source = "reconstructed_from_prepared_input"
    side_prediction = model.laterality_model.predict_array(native, source=path)
    side = side_prediction["laterality"]
    if side is None:
        raise ValueError("Side classifier cannot evaluate this prepared hip image")
    raw = model.predict_prepared(prepared, geometry, side)
    result = {"region": "hip", "source": str(path),
              "laterality": side, "laterality_prediction": side_prediction,
              "needs_review": bool(side_prediction["needs_review"]),
              "labels": {task: bool(raw["tasks"][short]["violation"]) for task, short in TASK_NAMES.items()},
              "scores": {task: raw["tasks"][short]["score"] for task, short in TASK_NAMES.items()},
              "tasks": {task: raw["tasks"][short] for task, short in TASK_NAMES.items()},
              "any_violation": bool(raw["any_violation"]),
              "metadata": {"laterality": side, "laterality_source": "model",
                           "laterality_confidence": side_prediction["confidence"],
                           "image_id": row["image_id"], "study_id": row["study_id"],
                           "checkpoints_dir": str(model.checkpoints_dir), "device": model.device,
                           "score_type": "five_fold_threshold_centered_mean",
                           "visualization_source": visualization_source, "note": raw["note"]},
              "geometry": _geometry(prepared, geometry, native, side, model.config),
              "image": image_payload(Image.fromarray(native))}
    result["annotated_image"] = annotate_image(result)
    return result


def run_all(data=DEFAULT_DATA_ROOT, checkpoints_dir=None, output_dir=DEFAULT_OUTPUT_DIR,
            *, device="auto", verbose=False, ci=False, train=False):
    """Run converted models only; optional training is explicitly requested."""
    data_root, output_dir = Path(data).resolve(), Path(output_dir).resolve()
    progress = Progress("hip", enabled=bool(verbose or ci))
    rows, labeled, assignments = _read_cohort(data_root)
    if train:
        from .pipeline import train as train_models
        train_models(data_root, checkpoints_dir=checkpoints_dir, device=device, verbose=verbose, ci=ci)
    model = load_checkpoints(checkpoints_dir, device=device, verbose=verbose, ci=ci)
    validation_path = model.runtime_dir / "export_validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        for filename, key in (("manifest.jsonl", "verification_manifest_sha256"), ("splits.json", "verification_splits_sha256")):
            expected = validation.get(key)
            if expected and hashlib.sha256((data_root / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"ONNX export and evaluation dataset disagree: {filename}")
    full_results = []
    for index, row in enumerate(rows, start=1):
        progress.update("infer", row["image_id"], current=index - 1, total=len(rows))
        path = data_root / row["image_path"]
        if data_root == DATA_ROOT.resolve():
            result = infer(path, model=model)
        else:
            result = _custom_prepared_result(data_root, row, model)
        full_results.append(result)
        progress.update("infer", row["image_id"], current=index, total=len(rows))
    oof, metrics = _evaluate_full(labeled, assignments, full_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, oof_path = output_dir / "predictions_full.json", output_dir / "oof_predictions.csv"
    metrics_path, report_path = output_dir / "metrics.json", output_dir / "report.md"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "cohort": {"all_hip_images": 155, "labeled_outer_fold_images": 150, "unlabeled_images": 5},
              "inference_runtime": "onnx_only",
              "metrics_scope": "150 eligible labeled images; original probability and threshold from one held-out outer fold per image",
              "evaluation_limitations": "Historical ROI early stopping and calibration reused validation labels; evaluation is not fully nested. Side is predicted by a separate model fitted on 118 development images; side was not cross-fitted within each quality fold. Five-fold deployment aggregate has no independent validation metric.",
              "metrics": metrics,
              "provenance": {"data_root": str(data_root), "checkpoints_dir": str(model.checkpoints_dir), "device": model.device,
                             "manifest_sha256": hashlib.sha256((data_root / "manifest.jsonl").read_bytes()).hexdigest(),
                             "splits_sha256": hashlib.sha256((data_root / "splits.json").read_bytes()).hexdigest()},
              "outputs": {"full_predictions": str(predictions_path), "oof_predictions": str(oof_path), "metrics": str(metrics_path), "report": str(report_path)}}
    write_json(predictions_path, full_results)
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)
    write_json(metrics_path, report)
    lines = ["# Hip ONNX evaluation", "", "All 155 images received deployment inference and ready visualizations.",
             "Metrics below use 150 labeled images and a single held-out outer fold for each image.", "",
             "| Task | F1 | ROC AUC | Balanced accuracy | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, values in metrics.items():
        lines.append(f"| {task} | {values['f1']:.6f} | {values['roc_auc']:.6f} | {values['balanced_accuracy']:.6f} | {values['tp']} | {values['fp']} | {values['fn']} | {values['tn']} |")
    lines.extend(["", report["evaluation_limitations"], "", "Inference imports no Torch, Transformers or scikit-learn."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress.update("metrics", f"Saved 155 predictions and 150 held-out scores to {output_dir}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoints-dir", "--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--train", action="store_true", help="Explicitly train and convert before ONNX inference")
    args = parser.parse_args(argv)
    run_all(args.data, args.checkpoints_dir, args.output_dir,
            device=args.device, verbose=args.verbose, ci=args.ci, train=args.train)


if __name__ == "__main__":
    main()
