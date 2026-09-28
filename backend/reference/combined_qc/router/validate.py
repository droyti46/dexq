"""Real integration validation of all 255 canonical DXA inputs.

This checks dispatch and output contracts, not independent quality-task
performance: production QC predictions are full inference, not OOF estimates.
No model is trained or converted by this script.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from PIL import Image
import numpy as np

from ..common import write_json
from .coordinator import QCPipeline

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DATA = ROOT.parent / "data/dxa_v1"
EXPECTED_TASKS = {
    "hip": {"hip_positioning_rotation", "hip_roi"},
    "spine": {"spine_positioning", "spine_axis", "spine_artifacts"},
}


def _progress(enabled, message):
    if enabled:
        print(f"[router validation] {message}", file=sys.stderr, flush=True)


def _identity(qc):
    """Capture every live reusable handle and every ONNX session identity."""
    classifier = qc.models["classifier"]
    hip = qc.models["hip"]
    spine = qc.models["spine"]
    return {
        "handles": {**{name: id(handle) for name, handle in sorted(qc.models.items())},
                    "hip_laterality": id(hip.laterality_model)},
        "sessions": {
            "classifier": {"resnet18": id(classifier.extractor.session)},
            "hip": {**{name: id(session) for name, session in sorted(hip.sessions.items())},
                    "laterality/resnet18": id(hip.laterality_model.extractor.session)},
            "spine": {"resnet18": id(spine.neural.resnet), "spinenet": id(spine.neural.spinenet)},
        },
    }


def _loaded_model_provenance(qc):
    """Capture input inventories once, independently of later publication."""
    classifier, hip, spine = (qc.models[name] for name in ("classifier", "hip", "spine"))
    spine_inventory_bytes = (spine.onnx_dir / "export_manifest.json").read_bytes()
    spine_manifest_matches = hashlib.sha256(spine_inventory_bytes).hexdigest() == spine.provenance["onnx_manifest_sha256"]
    return {
        "classifier": deepcopy(classifier.provenance),
        "hip": {"provenance": deepcopy(hip.provenance), "verified_files": deepcopy(hip.manifest["files"]),
                "laterality_provenance": deepcopy(hip.laterality_model.provenance)},
        "spine": {"provenance": deepcopy(spine.provenance),
                  "verified_onnx_inventory": json.loads(spine_inventory_bytes) if spine_manifest_matches else None,
                  "inventory_snapshot_matches_loaded_manifest": spine_manifest_matches},
    }


def _visualization(result):
    payload = result.get("annotated_image")
    if not isinstance(payload, dict) or payload.get("encoding") != "base64_png":
        raise AssertionError("A completed QC result must contain a ready base64 PNG")
    png = base64.b64decode(payload["data"], validate=True)
    with Image.open(BytesIO(png)) as image:
        if image.format != "PNG":
            raise AssertionError("Visualization is not a PNG")
        image.verify()
    with Image.open(BytesIO(png)) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    if (width, height) != (payload["width"], payload["height"]):
        raise AssertionError("Visualization payload dimensions do not match PNG")
    geometry = result["geometry"]
    native_width, native_height = geometry["image_width"], geometry["image_height"]
    if result["region"] == "hip":
        # Hip adds a caption panel beneath the complete unscaled native image.
        overlays = payload.get("overlays", {})
        if (overlays.get("native_image_width"), overlays.get("native_image_height")) != (native_width, native_height):
            raise AssertionError("Hip visualization native dimensions do not match geometry")
        if width < native_width or height < native_height:
            raise AssertionError("Hip visualization cropped the native image")
    elif (width, height) != (native_width, native_height):
        raise AssertionError("Spine visualization dimensions do not match native geometry")
    return {"width": width, "height": height, "mode": mode,
            "native_width": native_width, "native_height": native_height,
            "png_sha256": hashlib.sha256(png).hexdigest(), "png_bytes": len(png)}


def _assert_complete(result, target):
    if result["routing"]["target_module"] != target:
        raise AssertionError(f"Wrong target module: {result['routing']['target_module']}")
    if set(result["labels"]) != EXPECTED_TASKS[target]:
        raise AssertionError(f"Unexpected {target} task labels: {sorted(result['labels'])}")
    if set(result["scores"]) != EXPECTED_TASKS[target]:
        raise AssertionError("Score keys differ from task labels")
    if set(result["tasks"]) != EXPECTED_TASKS[target]:
        raise AssertionError("Task data keys differ from task labels")
    for label in result["labels"].values():
        if label is not None and not isinstance(label, (bool, np.bool_)):
            raise AssertionError("QC decisions must be booleans or None")
    if target == "hip":
        side = result["metadata"].get("laterality")
        if side not in {"left_hip", "right_hip"}:
            raise AssertionError("Automatic hip side prediction missing")
        if result["metadata"].get("laterality_source") != "model":
            raise AssertionError("Hip inference did not use its image-only side model")
        prediction = result.get("laterality_prediction", {})
        if result.get("laterality") != side or prediction.get("laterality") != side:
            raise AssertionError("Hip side prediction and metadata disagree")
        confidence = prediction.get("confidence")
        if (not isinstance(confidence, (int, float)) or not np.isfinite(confidence)
                or not .5 <= confidence <= 1 or result["metadata"].get("laterality_confidence") != confidence):
            raise AssertionError("Hip side confidence missing or malformed")
    return _visualization(result)


def _fresh_process(hip_source, spine_source):
    """Native library stdout/stderr are checked in a clean Python process."""
    code = (
        "import sys\n"
        "from combined_qc.router import QCPipeline\n"
        "qc=QCPipeline(verbose=False,ci=False)\n"
        f"hip=qc.infer({str(hip_source)!r},verbose=False,ci=False)\n"
        f"spine=qc.infer({str(spine_source)!r},verbose=False,ci=False)\n"
        "assert set(hip['labels'])=={'hip_positioning_rotation','hip_roi'}\n"
        "assert set(spine['labels'])=={'spine_positioning','spine_axis','spine_artifacts'}\n"
        "assert not any(name in sys.modules for name in ('torch','sklearn'))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT,
                            env=os.environ.copy(), capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or result.stdout or result.stderr:
        raise AssertionError(f"Fresh silent runtime failed: exit={result.returncode}, "
                             f"stdout={result.stdout[:1500]!r}, stderr={result.stderr[:1500]!r}")
    return {"passed": True, "stdout_bytes": 0, "stderr_bytes": 0,
            "torch_imported": False, "sklearn_imported": False}


def _current_runtime_equivalence(qc):
    """Allow provenance-only republishing, but reject changed live numerics."""
    classifier = qc.models["classifier"]
    directory = qc.checkpoint_dirs["classifier"] / "runtime"
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "region_logreg.npz", allow_pickle=False) as saved:
        identical = {key: bool(np.array_equal(getattr(classifier, key), saved[key]))
                     for key in ("mean", "scale", "coef", "intercept")}
        current_metadata = json.loads(str(saved["metadata"].item()))
    backbone_sha = hashlib.sha256((directory / "resnet18_features.onnx").read_bytes()).hexdigest()
    if (not all(identical.values()) or backbone_sha != classifier.metadata["backbone_sha256"]
            or backbone_sha != manifest["files"]["resnet18_features.onnx"]
            or any(current_metadata[key] != classifier.metadata[key]
                   for key in ("representation", "threshold", "review_threshold", "preprocessing"))):
        raise AssertionError("Current classifier runtime differs numerically from the validated live handle")
    side = qc.models["hip"].laterality_model
    side_directory = qc.checkpoint_dirs["hip"] / "runtime/laterality"
    side_manifest = json.loads((side_directory / "manifest.json").read_text())
    with np.load(side_directory / "laterality_logreg.npz", allow_pickle=False) as saved:
        side_identical = {key: bool(np.array_equal(getattr(side, key), saved[key]))
                          for key in ("mean", "scale", "coef", "intercept")}
    side_backbone_sha = hashlib.sha256((side_directory / "resnet18_features.onnx").read_bytes()).hexdigest()
    if not all(side_identical.values()) or side_backbone_sha != side.metadata["backbone_sha256"]:
        raise AssertionError("Current hip-side runtime differs numerically from the validated live handle")
    return {"passed": True, "numeric_arrays_identical": identical,
            "backbone_sha256_identical": True, "backbone_sha256": backbone_sha,
            "loaded_classifier_sha256": classifier.provenance["classifier_sha256"],
            "current_classifier_sha256": manifest["files"]["region_logreg.npz"],
            "metadata_identical": current_metadata == classifier.metadata,
            "hip_laterality_numeric_arrays_identical": side_identical,
            "hip_laterality_backbone_sha256": side_backbone_sha,
            "hip_laterality_loaded_classifier_sha256": side.provenance["classifier_sha256"],
            "hip_laterality_current_classifier_sha256": side_manifest["files"]["laterality_logreg.npz"]}


def _direct_equality(qc, representatives, sources, frozen_identity):
    from ..hip.pipeline import infer as hip_infer
    from ..spine.pipeline import infer as spine_infer
    checks = []
    for row in representatives:
        source = sources[row["image_id"]]
        target = "spine" if row["anatomical_region"] == "lumbar_spine" else "hip"
        routed = qc.infer(source)
        direct = (spine_infer if target == "spine" else hip_infer)(
            source, model=qc.models[target], device="auto")
        stripped = {key: value for key, value in routed.items() if key != "routing"}
        if stripped != direct:
            differing = sorted(key for key in set(stripped) | set(direct) if stripped.get(key) != direct.get(key))
            raise AssertionError(f"Routing changed child fields for {row['image_id']}: {differing}")
        repeated = qc.infer(source)
        if repeated != routed:
            raise AssertionError("Repeated routed inference changed its complete result")
        if _identity(qc) != frozen_identity:
            raise AssertionError("Repeated inference reloaded a model or ONNX session")
        checks.append({"image_id": row["image_id"], "target": target,
                       "child_result_equal_except_routing": True, "repeat_result_equal": True})
    return {"passed": True, "samples": checks}


def _automatic_side_and_folder(qc, hip_row, spine_row, sources, frozen_identity):
    from .preprocessing import load_native
    hip_pixels = load_native(sources[hip_row["image_id"]])
    spine_pixels = load_native(sources[spine_row["image_id"]])
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary).resolve()
        hip_path, spine_path, blank_path = (directory / name for name in ("a_hip.png", "b_spine.png", "c_blank.png"))
        Image.fromarray(hip_pixels).save(hip_path)
        Image.fromarray(spine_pixels).save(spine_path)
        Image.fromarray(np.zeros((48, 48), dtype=np.uint8)).save(blank_path)
        automatic = qc.infer(hip_path)
        _assert_complete(automatic, "hip")
        if automatic["metadata"]["laterality"] != hip_row["anatomical_region"]:
            raise AssertionError("Arbitrarily named native hip lost its automatic side prediction")
        mirrored_path = directory / "d_mirrored_hip.png"
        Image.fromarray(hip_pixels[:, ::-1]).save(mirrored_path)
        mirrored = qc.infer(mirrored_path)
        _assert_complete(mirrored, "hip")
        if mirrored["metadata"]["laterality"] == automatic["metadata"]["laterality"]:
            raise AssertionError("Horizontal mirror did not reverse the predicted image-side class")
        mirrored_path.unlink()
        with patch("combined_qc.hip.pipeline.infer", side_effect=AssertionError("Blank image reached hip QC")), \
                patch("combined_qc.spine.pipeline.infer", side_effect=AssertionError("Blank image reached spine QC")):
            blank = qc.infer(blank_path)
        if (blank.get("status") != "invalid_image" or blank["routing"]["target_module"] is not None
                or blank["labels"] or blank["any_violation"] is not None):
            raise AssertionError("Blank image must stay unavailable without QC")
        results = qc.infer(directory)
        repeated = qc.infer(directory)
        expected_sources = [str(hip_path), str(spine_path), str(blank_path)]
        for folder_result in (results, repeated):
            if [r["source"] for r in folder_result] != expected_sources:
                raise AssertionError("Folder failed to preserve all input files in sorted order")
            if [r["routing"]["target_module"] for r in folder_result] != ["hip", "spine", None]:
                raise AssertionError("Folder dispatched incorrect QC modules")
        _assert_complete(results[0], "hip")
        _assert_complete(results[1], "spine")
        if results != repeated:
            raise AssertionError("Repeated automatic folder inference changed its result")
        if results[2].get("status") != "invalid_image":
            raise AssertionError("Folder lost the blank image's unavailable status")
    if _identity(qc) != frozen_identity:
        raise AssertionError("Additional file/folder calls replaced reusable models")
    return {"passed": True, "arbitrarily_named_native_hip": "completed_with_automatic_side",
            "predicted_side": automatic["metadata"]["laterality"],
            "horizontal_mirror_side_prediction_reversed": True,
            "mirrored_predicted_side": mirrored["metadata"]["laterality"], "blank_image": "invalid_image",
            "blank_calls_quality_module": False, "folder_count_each_call": 3,
            "folder_statuses": ["completed", "completed", "invalid_image"]}


def run(*, output=None, ci=False):
    started = time.perf_counter()
    output = Path(output).resolve() if output else ROOT / "checkpoints/validation_report.json"
    runtime_manifest = ROOT / "checkpoints/runtime/manifest.json"
    if not runtime_manifest.is_file():
        raise FileNotFoundError("Router runtime has not been published; run its conversion before validation")
    starting_manifest_sha = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    manifest = DATA / "manifest.jsonl"
    rows = sorted((json.loads(line) for line in manifest.read_text().splitlines() if line.strip()), key=lambda r: r["image_id"])
    if len(rows) != 255 or len({row["image_id"] for row in rows}) != 255:
        raise AssertionError("Expected all 255 unique images, with no quality exclusion")
    sources = {row["image_id"]: (PROJECT / "source_data" / row["canonical_source_path"]).resolve() for row in rows}
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError("A canonical native DICOM is missing")
    representatives = [next(row for row in rows if row["anatomical_region"] == region)
                       for region in ("lumbar_spine", "left_hip", "right_hip")]
    checks = {}
    _progress(ci, "Checking fresh-process quiet inference and lazy dependencies")
    try:
        checks["fresh_process"] = _fresh_process(sources[representatives[1]["image_id"]], sources[representatives[0]["image_id"]])
    except Exception as error:
        checks["fresh_process"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    _progress(ci, "Loading one reusable QCPipeline and all converted model handles")
    qc = QCPipeline()
    qc.load_checkpoints()
    frozen_identity = _identity(qc)
    input_model_provenance = _loaded_model_provenance(qc)
    items = []
    classifications = Counter()
    routes = Counter()
    wrong = []
    side_wrong = []
    errors = []
    unrouted = []
    _progress(ci, "Running all 255 canonical native DICOMs")
    for index, row in enumerate(rows, 1):
        image_id = row["image_id"]
        expected_class = "lumbar_spine" if row["anatomical_region"] == "lumbar_spine" else "hip"
        target = "spine" if expected_class == "lumbar_spine" else "hip"
        try:
            result = qc.infer(sources[image_id])
            classification = result["routing"].get("region")
            classifications[str(classification)] += 1
            selected = result["routing"].get("target_module")
            routes[str(selected)] += 1
            if classification != expected_class:
                wrong.append(image_id)
            if target == "hip" and result.get("metadata", {}).get("laterality") != row["anatomical_region"]:
                side_wrong.append(image_id)
            completed = set(result.get("labels", {})) == EXPECTED_TASKS[target] and result.get("annotated_image") is not None
            if not completed:
                unrouted.append(image_id)
            visualization = _assert_complete(result, target) if completed else None
            if _identity(qc) != frozen_identity:
                raise AssertionError("Model handle/session identity changed during inference")
            items.append({"image_id": image_id, "expected_anatomical_region": row["anatomical_region"],
                          "routing_region": classification, "target_module": selected,
                          "routing_confidence": result["routing"].get("confidence"),
                          "status": result.get("status", "completed"),
                          "profile_label_keys": sorted(result.get("labels", {})),
                          "labels": result.get("labels", {}), "any_violation": result.get("any_violation"),
                          "hip_laterality": result.get("metadata", {}).get("laterality"),
                          "hip_laterality_source": result.get("metadata", {}).get("laterality_source"),
                          "hip_laterality_confidence": result.get("metadata", {}).get("laterality_confidence"),
                          "needs_review": result.get("needs_review", False),
                          "visualization": visualization})
        except Exception as error:
            failure = {"image_id": image_id, "error": f"{type(error).__name__}: {error}"}
            errors.append(failure)
            items.append({**failure, "expected_anatomical_region": row["anatomical_region"], "status": "error"})
            if len(errors) <= 3:
                _progress(ci, f"Exception for {image_id}: {failure['error']}")
        if index % 30 == 0 or index == len(rows):
            _progress(ci, f"Processed {index}/{len(rows)}, exceptions={len(errors)}, classification_errors={len(wrong)}, unavailable={len(unrouted)}")
    for name, operation in (
        ("direct_child_equality_and_repeats", lambda: _direct_equality(qc, representatives, sources, frozen_identity)),
        ("automatic_hip_side_and_mixed_folder", lambda: _automatic_side_and_folder(qc, representatives[1], representatives[0], sources, frozen_identity)),
        ("current_runtime_numeric_equivalence", lambda: _current_runtime_equivalence(qc)),
    ):
        _progress(ci, f"Checking {name}")
        try:
            checks[name] = operation()
        except Exception as error:
            checks[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    passed = not (wrong or side_wrong or errors or unrouted) and all(check.get("passed") for check in checks.values())
    report = {
        "passed": passed, "protocol": "Full canonical-native integration inference, not OOF quality evaluation",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "router_runtime_manifest_sha256_at_start": starting_manifest_sha,
        "router_runtime_manifest_sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
        "n_inputs": len(rows), "quality_exclusions": 0,
        "quality_ineligible_included": sum(not row["eligible"] for row in rows),
        "expected_regions": dict(Counter("lumbar_spine" if row["anatomical_region"] == "lumbar_spine" else "hip" for row in rows)),
        "classified_regions": dict(classifications), "routed_modules": dict(routes),
        "classification_error_count": len(wrong), "classification_error_ids": wrong,
        "hip_laterality_error_count": len(side_wrong), "hip_laterality_error_ids": side_wrong,
        "unrouted_count": len(unrouted), "unrouted_ids": unrouted,
        "exception_count": len(errors), "errors": errors,
        "model_handle_identity_preserved": _identity(qc) == frozen_identity,
        "onnx_session_count": sum(len(sessions) for sessions in frozen_identity["sessions"].values()),
        "numeric_classifier_count": 3, "model_identities": frozen_identity,
        "input_model_provenance": input_model_provenance,
        "checks": checks, "per_image": items, "seconds": time.perf_counter() - started,
        "limitations": ["This mixes training and held-out anatomy images; it is an integration consistency check",
                        "No independent F1/AUC for quality subtasks is estimated by these full-inference predictions"],
    }
    write_json(output, report)
    _progress(ci, f"Report saved to {output}; passed={passed}")
    if not passed:
        raise AssertionError(f"Router integration validation failed; inspect {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", action="store_true", help="Print bounded integration progress to stderr")
    parser.add_argument("--output", type=Path, help="Default: router/checkpoints/validation_report.json")
    args = parser.parse_args()
    run(output=args.output, ci=args.ci)


if __name__ == "__main__":
    main()
