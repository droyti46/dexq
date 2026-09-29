"""Measure current deployment behavior on every canonical native DXA image.

Run with ``python -m combined_qc.report_metrics --ci``. The same persistent
QCPipeline used by an application performs automatic anatomy/side prediction
and complete ONNX/NPZ quality inference. Nothing is fitted, converted, tuned,
or selected here. These full-dataset descriptive metrics include development
and training images; they are not an independent generalization estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .common import write_json
from .router.coordinator import QCPipeline

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MANIFEST = ROOT / "data/dxa_v1/manifest.jsonl"
SPINE_TRUTH = ROOT / "spine/data/labeled_manifest.jsonl"
TASKS = {
    "hip": ("hip_positioning_rotation", "hip_roi"),
    "spine": ("spine_positioning", "spine_axis", "spine_artifacts"),
}
SCORE_SEMANTICS = {
    "hip_positioning_rotation": "single logistic regression on frozen ResNet18 features; threshold-centered score 0.5; uncalibrated",
    "hip_roi": "single logistic regression on frozen ResNet18 features; threshold-centered score 0.5; uncalibrated",
    "spine_positioning": "maximum standardized coverage-violation margin; threshold 0; not a probability",
    "spine_axis": "absolute native-coordinate angle in degrees; threshold 5; not a probability",
    "spine_artifacts": "single logistic-regression violation score; threshold 0.5; uncalibrated",
}


def _progress(enabled, message):
    if enabled:
        print(f"[deployment metrics] {message}", file=sys.stderr, flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image IDs in {path}")
    return rows


def _binary(value, *, name):
    if value is None:
        return None
    if not isinstance(value, (bool, int)) or value not in (0, 1):
        raise ValueError(f"Expected a Boolean/binary value for {name}, got {value!r}")
    return bool(value)


def _score(value, *, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite inference score for {name}, got {value!r}")
    return float(value)


def _truth_rows(rows, corrected):
    """Use corrected expert criterion columns for spine, shared truth for hip."""
    spine = {row["image_id"]: row for row in corrected}
    shared_spine_ids = {row["image_id"] for row in rows if row["anatomical_region"] == "lumbar_spine"}
    if set(spine) != shared_spine_ids:
        raise ValueError("Corrected spine truth does not cover exactly the 100 shared spine IDs")
    truth = {}
    for row in rows:
        region = "spine" if row["anatomical_region"] == "lumbar_spine" else "hip"
        selected = spine[row["image_id"]] if region == "spine" else row
        labels = {task: _binary(selected["targets"][task], name=f"truth {task}") for task in TASKS[region]}
        quality = _binary(selected["quality_label"], name="truth quality_label")
        if all(value is not None for value in labels.values()) and quality != any(labels.values()):
            raise ValueError(f"Criterion truth and OR truth disagree for {row['image_id']}")
        truth[row["image_id"]] = {"region": region, "labels": labels, "any_violation": quality,
                                  "laterality": row["anatomical_region"] if region == "hip" else None,
                                  "source": "corrected_spine_expert_criterion_columns" if region == "spine" else "shared_hip_manifest"}
    return truth


def _auc(truth, scores):
    """Exact pairwise binary ROC AUC, including a half credit for tied scores."""
    positive = [score for label, score in zip(truth, scores) if label]
    negative = [score for label, score in zip(truth, scores) if not label]
    if not positive or not negative:
        return None
    return sum(1 if pos > neg else .5 if pos == neg else 0
               for pos in positive for neg in negative) / (len(positive) * len(negative))


def _metrics(items, *, truth_key, prediction_key, score_key=None, score_semantics=None):
    """Never silently omit an unavailable prediction with known truth."""
    known = [item for item in items if item.get(truth_key) is not None]
    missing_labels = [item["image_id"] for item in known if item.get(prediction_key) is None]
    missing_scores = ([item["image_id"] for item in known if item.get(score_key) is None]
                      if score_key is not None else [])
    result = {"n_inputs": len(items), "n_labeled": len(known), "n_unlabeled": len(items) - len(known),
              "n_positive": sum(bool(item[truth_key]) for item in known),
              "unknown_prediction_count": len(missing_labels), "unknown_prediction_ids": missing_labels,
              "unknown_score_count": len(missing_scores), "unknown_score_ids": missing_scores,
              "positive_class": "quality violation", "accuracy": None, "f1": None,
              "macro_f1": None, "balanced_accuracy": None, "roc_auc": None, "confusion": None}
    if score_semantics is not None:
        result["score_semantics"] = score_semantics
    if score_key is None:
        result["roc_auc_reason"] = "Production OR returns a Boolean decision only; no shared continuous OR score is defined"
    elif missing_scores:
        result["roc_auc_reason"] = "Unavailable score(s) with known truth; no scores silently excluded"
    if missing_labels or not known:
        result["status"] = "incomplete_predictions" if missing_labels else "no_truth"
        return result
    tp = sum(bool(item[truth_key]) and bool(item[prediction_key]) for item in known)
    fp = sum(not bool(item[truth_key]) and bool(item[prediction_key]) for item in known)
    fn = sum(bool(item[truth_key]) and not bool(item[prediction_key]) for item in known)
    tn = len(known) - tp - fp - fn
    f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    result.update(status="complete", confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                  accuracy=(tp + tn) / len(known), f1=f1_positive,
                  macro_f1=(f1_positive + f1_negative) / 2,
                  balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2
                  if tp + fn and tn + fp else None)
    if score_key is not None and not missing_scores:
        result["roc_auc"] = _auc([bool(item[truth_key]) for item in known], [item[score_key] for item in known])
    return result


def _runtime_inventory(qc):
    inventories = {}
    for name, directory in sorted(qc.checkpoint_dirs.items()):
        runtime = directory / "runtime"
        inventories[name] = {"directory": str(runtime), "files": {
            str(path.relative_to(runtime)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(runtime.rglob("*")) if path.is_file()}}
    return inventories


def _identities(qc):
    classifier, hip, spine = (qc.models[name] for name in ("classifier", "hip", "spine"))
    return {"handles": {**{name: id(handle) for name, handle in qc.models.items()},
                        "hip_laterality": id(hip.laterality_model)},
            "sessions": {"classifier": [id(classifier.extractor.session)],
                         "hip": [id(session) for _, session in sorted(hip.sessions.items())] + [id(hip.laterality_model.extractor.session)],
                         "spine": [id(spine.neural.resnet), id(spine.neural.spinenet)]}}


def _lean_result(row, source, truth, result):
    routing = result.get("routing", {})
    side = result.get("laterality_prediction") or {}
    labels = {task: _binary(value, name=task) for task, value in result.get("labels", {}).items()}
    scores = {task: _score(value, name=task) for task, value in result.get("scores", {}).items()}
    any_violation = _binary(result.get("any_violation"), name="any_violation")
    selected = routing.get("target_module")
    if selected in TASKS and (set(labels) != set(TASKS[selected]) or set(scores) != set(TASKS[selected])):
        raise ValueError("Incomplete or unexpected selected-profile label/score keys")
    if labels and all(value is not None for value in labels.values()) and any_violation != any(labels.values()):
        raise ValueError("Inference OR differs from OR of its actual task labels")
    routing_scores = routing.get("scores") or {}
    side_scores = side.get("scores") or {}
    return {"image_id": row["image_id"], "study_id": row["study_id"],
            "source": str(source), "source_sha256": _sha256(source),
            "declared_native_sha256": row["native_sha256"],
            "expected_region": truth["region"], "truth": truth,
            "status": result.get("status", "completed"), "labels": labels, "scores": scores,
            "any_violation": any_violation, "needs_review": bool(result.get("needs_review", False)),
            "routing": {"region": routing.get("region"), "target_module": selected,
                        "confidence": routing.get("confidence"), "needs_review": routing.get("needs_review"),
                        "scores": {key: _score(value, name=f"routing {key}") for key, value in routing_scores.items()}},
            "laterality": result.get("laterality"), "laterality_source": result.get("metadata", {}).get("laterality_source"),
            "laterality_prediction": {"laterality": side.get("laterality"), "confidence": side.get("confidence"),
                                      "needs_review": side.get("needs_review"),
                                      "scores": {key: _score(value, name=f"laterality {key}") for key, value in side_scores.items()}},
            "ready_visualization_returned": isinstance(result.get("annotated_image"), dict)}


def _comparison(items):
    """Prior integration labels are a check only, never an inference input."""
    path = ROOT / "router/checkpoints/validation_report.json"
    if not path.is_file():
        return {"available": False}
    old = json.loads(path.read_text(encoding="utf-8"))
    previous = {item["image_id"]: item for item in old.get("per_image", [])}
    changed, absent = [], []
    for item in items:
        reference = previous.get(item["image_id"])
        if reference is None:
            absent.append(item["image_id"])
        elif any(item.get(key) != reference.get(key) for key in ("labels", "any_violation")):
            changed.append(item["image_id"])
    return {"available": True, "path": str(path), "sha256": _sha256(path),
            "compared_count": len(items) - len(absent), "missing_ids": absent,
            "changed_quality_prediction_ids": changed,
            "all_quality_labels_identical": not absent and not changed}


def run(*, output=None, ci=False, router_checkpoints=None, hip_checkpoints=None, spine_checkpoints=None):
    started = time.perf_counter()
    output = Path(output).expanduser().resolve() if output else ROOT / "outputs/deployment_metrics.json"
    rows = sorted(_read_rows(MANIFEST), key=lambda row: row["image_id"])
    truth = _truth_rows(rows, _read_rows(SPINE_TRUTH))
    if len(rows) != 255 or Counter(item["region"] for item in truth.values()) != {"hip": 155, "spine": 100}:
        raise ValueError("Expected all 255 unique canonical images: 155 hip and 100 spine")
    sources = {row["image_id"]: (PROJECT / "source_data" / row["canonical_source_path"]).resolve() for row in rows}
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError("A canonical native image is missing; no inputs may be excluded")
    _progress(ci, "Loading one persistent QCPipeline; training/conversion disabled")
    qc = QCPipeline(router_checkpoints, hip_checkpoints_dir=hip_checkpoints,
                    spine_checkpoints_dir=spine_checkpoints, verbose=False, ci=False).load_checkpoints()
    identities = _identities(qc)
    inventory = _runtime_inventory(qc)
    provenance = {name: deepcopy(handle.provenance) for name, handle in qc.models.items()}
    provenance["hip_laterality"] = deepcopy(qc.models["hip"].laterality_model.provenance)
    items, errors = [], []
    _progress(ci, "Running the exact public inference API on all 255 native DICOMs")
    for index, row in enumerate(rows, 1):
        image_id = row["image_id"]
        try:
            result = qc.infer(sources[image_id], verbose=False, ci=False)
            item = _lean_result(row, sources[image_id], truth[image_id], result)
            if _identities(qc) != identities:
                raise AssertionError("Inference replaced a persistent model handle or ONNX session")
        except Exception as error:
            failure = {"image_id": image_id, "error": f"{type(error).__name__}: {error}"}
            errors.append(failure)
            item = {**failure, "expected_region": truth[image_id]["region"], "truth": truth[image_id],
                    "status": "error", "labels": {}, "scores": {}, "any_violation": None,
                    "routing": {}, "laterality_prediction": {}}
            _progress(ci, f"Exception for {image_id}: {failure['error']}")
        items.append(item)
        if index % 30 == 0 or index == len(rows):
            _progress(ci, f"Processed {index}/{len(rows)}, errors={len(errors)}")
    regions = {}
    for region, tasks in TASKS.items():
        selected = [item for item in items if item["expected_region"] == region]
        flattened = [{"image_id": item["image_id"], "true": item["truth"]["any_violation"],
                      "prediction": item.get("any_violation")} for item in selected]
        regions[region] = {"n_processed": len(selected),
                           "or": _metrics(flattened, truth_key="true", prediction_key="prediction"), "tasks": {}}
        for task in tasks:
            task_items = [{"image_id": item["image_id"], "true": item["truth"]["labels"][task],
                           "prediction": item.get("labels", {}).get(task),
                           "score": item.get("scores", {}).get(task)} for item in selected]
            regions[region]["tasks"][task] = _metrics(task_items, truth_key="true", prediction_key="prediction",
                                                     score_key="score", score_semantics=SCORE_SEMANTICS[task])
    pooled = [{"image_id": item["image_id"], "true": item["truth"]["any_violation"],
               "prediction": item.get("any_violation")} for item in items]
    anatomy = [{"image_id": item["image_id"], "true": item["expected_region"] == "spine",
                "prediction": None if item.get("routing", {}).get("region") is None else item["routing"]["region"] == "lumbar_spine",
                "score": item.get("routing", {}).get("scores", {}).get("lumbar_spine")} for item in items]
    side = [{"image_id": item["image_id"], "true": item["truth"]["laterality"] == "right_hip",
             "prediction": None if item.get("laterality") is None else item["laterality"] == "right_hip",
             "score": item.get("laterality_prediction", {}).get("scores", {}).get("right_hip")}
            for item in items if item["expected_region"] == "hip"]
    anatomy_metrics = _metrics(anatomy, truth_key="true", prediction_key="prediction", score_key="score",
                               score_semantics="actual routing.scores.lumbar_spine; uncalibrated")
    anatomy_metrics["positive_class"] = "lumbar_spine"
    side_metrics = _metrics(side, truth_key="true", prediction_key="prediction", score_key="score",
                            score_semantics="actual laterality_prediction.scores.right_hip; uncalibrated")
    side_metrics.update(positive_class="right_hip", label_status="provisional_visual; verified anatomical side is unavailable")
    unknown = [item["image_id"] for item in items if item.get("any_violation") is None
               or any(item.get("labels", {}).get(task) is None or item.get("scores", {}).get(task) is None
                      for task in TASKS[item["expected_region"]])]
    unrouted = [item["image_id"] for item in items if item.get("routing", {}).get("target_module") is None]
    route_errors = [item["image_id"] for item in items if item.get("routing", {}).get("target_module") != item["expected_region"]]
    runtime_unchanged = _runtime_inventory(qc) == inventory
    handles_unchanged = _identities(qc) == identities
    comparison = _comparison(items)
    passed = not errors and not unknown and not unrouted and runtime_unchanged and handles_unchanged
    report = {
        "format": "combined_qc_deployment_metrics_v1", "passed": passed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Actual QCPipeline.infer on all canonical native DICOMs with automatic routing and hip side; identical full deployment rule for every metric",
        "evaluation_scope": "Full-dataset descriptive inference; includes training/development images, not OOF and not an independent generalization estimate",
        "positive_class": "True = quality violation", "n_processed": len(items),
        "n_labeled": sum(item["any_violation"] is not None for item in truth.values()),
        "n_unlabeled": sum(item["any_violation"] is None for item in truth.values()),
        "unlabeled_ids": [image_id for image_id, item in truth.items() if item["any_violation"] is None],
        "quality_exclusions": 0, "quality_ineligible_included": sum(not row["eligible"] for row in rows),
        "exception_count": len(errors), "errors": errors, "unknown_prediction_count": len(unknown),
        "unknown_prediction_ids": unknown, "unrouted_count": len(unrouted), "unrouted_ids": unrouted,
        "routing_error_count": len(route_errors), "routing_error_ids": route_errors,
        "model_handle_identity_preserved": handles_unchanged, "runtime_files_unchanged": runtime_unchanged,
        "onnx_session_count": sum(len(value) for value in identities["sessions"].values()),
        "numeric_classifier_count": sum(path.endswith(".npz") for region in inventory.values() for path in region["files"]),
        "provenance": {"dataset_manifest": str(MANIFEST), "dataset_manifest_sha256": _sha256(MANIFEST),
                       "corrected_spine_truth": str(SPINE_TRUTH), "corrected_spine_truth_sha256": _sha256(SPINE_TRUTH),
                       "truth_rule": "Shared hip labels150; corrected expert-criterion spine labels99 override shared spine labels96",
                       "model_handles": provenance, "runtime_file_inventory": inventory,
                       "training_performed": False, "conversion_performed": False, "threshold_tuning_performed": False},
        "overall_or": _metrics(pooled, truth_key="true", prediction_key="prediction"),
        "regions": regions, "auxiliary_classifiers": {"anatomy": anatomy_metrics, "hip_laterality": side_metrics},
        "comparison_to_previous_integration": comparison,
        "limitations": [
            "All tables use current full inference; quality models/thresholds have seen these development data",
            "Hip quality uses one frozen ResNet18 and two final LRs fitted on all150 labels; grouped validation trains separate temporary models",
            "The single spine artifact classifier was fitted on all99 labeled spine images",
            "Hip side truth is provisional; perfect agreement does not verify patient anatomical side",
            "OR has no continuous production score: its ROC AUC is intentionally null; heterogeneous task scores are not combined into an invented probability",
        ],
        "per_image": items, "seconds": time.perf_counter() - started,
    }
    write_json(output, report)
    _progress(ci, f"Saved {output}; passed={passed}; labeled={report['n_labeled']}/{len(items)}")
    if not passed:
        raise RuntimeError(f"Deployment evaluation is incomplete or runtime changed; inspect {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", action="store_true", help="Print bounded evaluation progress to stderr")
    parser.add_argument("--output", type=Path, help="Default: combined_qc/outputs/deployment_metrics.json")
    args = parser.parse_args()
    run(output=args.output, ci=args.ci)


if __name__ == "__main__":
    main()
