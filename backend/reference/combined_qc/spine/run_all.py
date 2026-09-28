"""Run the deployed single-model spine pipeline on all 100 bundled images.

The artifact logistic regression is loaded once from its numeric NPZ. This
script does not train models or perform cross-validation. It writes predictions
for every image; artifact metrics are calculated only for labeled images absent
from the model's recorded training set. The other criteria and OR are not
reported as independently evaluated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .engine.utils import read_manifest, sha256
from .pipeline import SPINE_TASKS, load_checkpoints
from ..common import Progress
from .portable_inference import PortableSpinePipeline


ROOT = Path(__file__).resolve().parent
TASKS = (*SPINE_TASKS, "spine_or")
DISPLAY_NAMES = {
    "spine_positioning": "Охват",
    "spine_axis": "Угол",
    "spine_artifacts": "Артефакты",
    "spine_or": "Любое нарушение (OR)",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=_json_default) + "\n",
                    encoding="utf-8")


def _geometry(result: dict) -> dict:
    """Expose native-pixel drawing data in the same shape as ``spine.infer``."""
    landmarks, axis = result["shared_landmarks"], result["axis"]
    sequence = result["coverage"]["components"]["upper_gap_sequence"]
    top, bottom = axis.get("top_xy"), axis.get("bottom_xy")
    axis_line = None if top is None or bottom is None else {
        "top_xy": top,
        "bottom_xy": bottom,
        "angle_deg": axis.get("angle_deg"),
        "signed_angle_deg": axis.get("signed_angle_deg"),
    }
    return {
        "coordinate_system": landmarks["coordinate_system"],
        "image_width": landmarks["image_width"],
        "image_height": landmarks["image_height"],
        "axis_line": axis_line,
        "vertebral_candidates": landmarks["vertebral_candidates"],
        "gap_lines": [
            {"center_xy": [gap["x"], gap["y"]],
             "endpoints_xy": gap["endpoints"], "contrast": gap["contrast"]}
            for gap in sequence["observed_gaps"]
        ],
        "pelvic_proxies": sequence["pelvic_proxies"],
    }


def _model_membership(root: Path, images: list[dict], labels: list[dict]) -> tuple[dict, set[str]]:
    """Validate that saved model provenance names exactly its fitting images."""
    path = root / "checkpoints/runtime/classifiers/model_manifest.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    if (spec.get("format") != "spine_single_artifact_model_v1"
            or spec.get("n_models_trained") != 1
            or spec.get("model_file") != "artifact_logreg.npz"):
        raise ValueError("Unexpected single-model manifest")
    if sha256(root / "checkpoints/runtime/classifiers/artifact_logreg.npz") != spec.get("model_sha256"):
        raise ValueError("Artifact model checksum differs from its manifest")
    if sha256(root / "checkpoints/runtime/onnx/resnet18_features.onnx") != spec.get("onnx_resnet_sha256"):
        raise ValueError("ResNet18 ONNX differs from the classifier training extractor")
    if sha256(root / "data/manifest.jsonl") != spec.get("image_manifest_sha256"):
        raise ValueError("Image manifest differs from classifier training data")
    if sha256(root / "data/labeled_manifest.jsonl") != spec.get("corrected_labels_sha256"):
        raise ValueError("Corrected labels differ from classifier training data")
    image_ids = {row["image_id"] for row in images}
    label_by_id = {row["image_id"]: row for row in labels}
    if (len(images) != 100 or len(image_ids) != 100 or len(labels) != 100
            or len(label_by_id) != 100 or set(label_by_id) != image_ids):
        raise ValueError("Expected the aligned set of 100 spine images and labels")
    training_ids = spec.get("train_image_ids")
    if (not isinstance(training_ids, list)
            or len(training_ids) != len(set(training_ids))
            or not set(training_ids) <= image_ids
            or len(training_ids) != spec.get("n_train")):
        raise ValueError("Invalid artifact training membership")
    if any(label_by_id[image_id]["targets"]["spine_artifacts"] not in (0, 1)
           for image_id in training_ids):
        raise ValueError("Artifact training membership includes unlabeled images")
    n_positive = sum(label_by_id[image_id]["targets"]["spine_artifacts"]
                     for image_id in training_ids)
    if n_positive != spec.get("n_positive"):
        raise ValueError("Saved artifact training prevalence differs from labels")
    return spec, set(training_ids)


def _artifact_metrics(rows: list[dict]) -> dict:
    """Metrics only for artifact cases excluded from fitting this exact NPZ."""
    held_out = [row for row in rows if row["artifact_evaluation_role"] == "held_out_from_fit"]
    measured = [row for row in held_out if row["spine_artifacts_pred"] is not None]
    result = {
        "status": ("available_held_out_from_fit" if measured else
                   "unavailable_no_predictions" if held_out else
                   "unavailable_no_held_out_images"),
        "n": len(held_out), "n_predictions": len(measured),
        "n_unknown": len(held_out) - len(measured),
        "positives": sum(row["spine_artifacts_true"] == 1 for row in held_out),
        "threshold": 0.5,
        "f1": None, "roc_auc": None, "balanced_accuracy": None,
        "accuracy": None, "precision": None, "recall": None,
        "specificity": None, "tp": None, "fn": None, "fp": None, "tn": None,
    }
    if not held_out or not measured:
        return result
    tp = sum(row["spine_artifacts_true"] == 1 and row["spine_artifacts_pred"] is True
             for row in measured)
    fn = sum(row["spine_artifacts_true"] == 1 and row["spine_artifacts_pred"] is False
             for row in measured)
    fp = sum(row["spine_artifacts_true"] == 0 and row["spine_artifacts_pred"] is True
             for row in measured)
    tn = sum(row["spine_artifacts_true"] == 0 and row["spine_artifacts_pred"] is False
             for row in measured)
    positive_scores = [row["spine_artifacts_score"] for row in measured
                       if row["spine_artifacts_true"] == 1]
    negative_scores = [row["spine_artifacts_score"] for row in measured
                       if row["spine_artifacts_true"] == 0]
    roc_auc = None
    if positive_scores and negative_scores:
        roc_auc = sum((a > b) + 0.5 * (a == b)
                      for a in positive_scores for b in negative_scores) / (
                          len(positive_scores) * len(negative_scores))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    result.update({
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": roc_auc,
        "balanced_accuracy": ((sensitivity + specificity) / 2
                              if sensitivity is not None and specificity is not None else None),
        "accuracy": (tp + tn) / len(measured),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": sensitivity,
        "specificity": specificity,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    })
    return result


def _write_table(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, *, device: str = "auto", verbose=False, ci=False) -> dict:
    """Infer all bundled images without fitting or cross-validation."""
    if device not in ("auto", "cpu"):
        raise ValueError("This ONNX Runtime pipeline uses CPU; choose --device cpu or auto")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress = Progress("spine", verbose or ci)

    images = read_manifest(ROOT / "data/manifest.jsonl")
    labels = read_manifest(ROOT / "data/labeled_manifest.jsonl")
    model_spec, training_ids = _model_membership(ROOT, images, labels)
    label_by_id = {row["image_id"]: row for row in labels}
    model = load_checkpoints(device=device, verbose=verbose, ci=ci)
    if model.provenance["classifier_binary_sha256"] != model_spec["model_sha256"]:
        raise ValueError("Loaded classifier differs from saved training manifest")
    inference_started = time.perf_counter()
    progress.update("infer", "All bundled spine images", 0, len(images))
    predictions = model.predict_records(images, ROOT / "data", mode="full")
    progress.update("infer", "All bundled spine images", len(predictions), len(images))
    inference_seconds = time.perf_counter() - inference_started
    if (len(predictions) != 100
            or [row["image_id"] for row in images] != [row["image_id"] for row in predictions]
            or any(row.get("classifier_variant") != "single" for row in predictions)):
        raise ValueError("Single-model inference did not return all 100 spine images in order")

    full_results, table_rows = [], []
    for image, prediction in zip(images, predictions):
        image_id = image["image_id"]
        result = prediction["result"]
        truth = label_by_id[image_id]
        if truth["study_id"] != image["study_id"]:
            raise ValueError(f"Image and label study IDs disagree: {image_id}")
        artifact_truth = truth["targets"]["spine_artifacts"]
        role = ("unlabeled" if artifact_truth is None else
                "training" if image_id in training_ids else "held_out_from_fit")
        decisions = {task: result[part]["violation"] for task, part in SPINE_TASKS.items()}
        decisions["spine_or"] = result["quality"]["violation"]
        scores = {task: result[part]["score"] for task, part in SPINE_TASKS.items()}
        full_results.append({
            "image_id": image_id,
            "view_id": image["legacy_view_id"],
            "study_id": image["study_id"],
            "native_path": image["native_path"],
            "artifact_evaluation_role": role,
            "classifier_variant": "single",
            "labels": decisions,
            "scores": scores,
            "geometry": _geometry(result),
            "result": result,
        })
        row = {
            "image_id": image_id, "view_id": image["legacy_view_id"],
            "study_id": image["study_id"], "artifact_evaluation_role": role,
        }
        for task in SPINE_TASKS:
            row[task + "_true"] = truth["targets"][task]
            row[task + "_pred"] = decisions[task]
            row[task + "_score"] = scores[task]
        row["spine_or_true"] = truth["quality_label"]
        row["spine_or_pred"] = decisions["spine_or"]
        table_rows.append(row)

    _write_json(output_dir / "predictions_full.json", full_results)
    _write_table(output_dir / "predictions_table.csv", table_rows,
                 ["image_id", "view_id", "study_id", "artifact_evaluation_role",
                  *(key for task in SPINE_TASKS for key in
                    (task + "_true", task + "_pred", task + "_score")),
                  "spine_or_true", "spine_or_pred"])

    artifact = _artifact_metrics(table_rows)
    unavailable = {"status": "not_independently_evaluated", "n": 0,
                   "f1": None, "roc_auc": None, "balanced_accuracy": None}
    metrics = {
        "evaluation_protocol": "one artifact logistic regression, full-mode inference on all 100 images",
        "n_images": len(images),
        "n_labeled": sum(row["targets"]["spine_artifacts"] is not None for row in labels),
        "n_artifact_training_images": len(training_ids),
        "n_artifact_held_out_images": artifact["n"],
        "independent_artifact_metrics_available": artifact["n_predictions"] > 0,
        "metric_scope": ("Artifact metrics use only labeled images excluded from fitting "
                         "this NPZ. No coverage, axis, or OR performance is measured by "
                         "this run; the corresponding previous rules/models may have "
                         "been developed on these images."),
        "spine_positioning": dict(unavailable),
        "spine_axis": dict(unavailable),
        "spine_artifacts": artifact,
        "spine_or": dict(unavailable),
    }
    _write_json(output_dir / "metrics.json", metrics)
    metric_rows = []
    for task in TASKS:
        item = metrics[task]
        metric_rows.append({
            "Проверка": DISPLAY_NAMES[task], "Статус": item["status"],
            "N": item["n"], "F1": item["f1"], "ROC AUC": item["roc_auc"],
            "Balanced accuracy": item["balanced_accuracy"],
        })
    _write_table(output_dir / "metrics.csv", metric_rows,
                 ["Проверка", "Статус", "N", "F1", "ROC AUC", "Balanced accuracy"])

    report = {
        "n_images": len(full_results),
        "n_labeled": metrics["n_labeled"],
        "device": "cpu_onnxruntime",
        "artifact_classifiers_loaded": 1,
        "artifact_model_sha256": model_spec["model_sha256"],
        "artifact_model_train_cases": len(training_ids),
        "artifact_model_held_out_cases": artifact["n"],
        "prediction_mode": "single_model_full_for_all_100",
        "training_performed": False,
        "independent_artifact_metrics_available": artifact["n_predictions"] > 0,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "files": ["predictions_full.json", "predictions_table.csv",
                  "metrics.json", "metrics.csv", "run_report.json"],
    }
    _write_json(output_dir / "run_report.json", report)
    progress.update("metrics", f"Saved {len(full_results)} predictions to {output_dir}; "
                    f"artifact held-out cases: {artifact['n']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/current_run")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto",
                        help="ONNX Runtime inference currently uses CPU")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, device=args.device, verbose=args.verbose, ci=args.ci)


if __name__ == "__main__":
    main()
