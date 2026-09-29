"""Convert compact hip source weights into a verified ONNX/numeric bundle.

One frozen public quality graph feeds the positioning and ROI logistic models.
The separately trained automatic side pair is exported into the same bundle.
Conversion never fits models and publishes atomically after parity succeeds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from combined_qc.common import Progress, publish_directory, quiet_output, resolve_checkpoint_paths, write_json
from .preprocessing import acquisition_features, letterbox, sha256
from .runtime import (
    ENCODER_NAME, FEATURE_SPECS, FEATURE_STATE_SHA, FORMAT, OUTPUT_SPECS,
    PUBLIC_SOURCE_SHA, SOURCE_FORMAT, TASKS, HipONNXPredictor,
    load_numeric_classifier, score_numeric,
)

GEOMETRY_FEATURE_NAMES = (
    "superior_joint_intensity_ratio", "joint_field_intensity_ratio",
    "inferior_shaft_intensity_ratio", "inferior_shaft_foreground_fraction",
    "lateral_bone_intensity_max_ratio", "lateral_intensity_asymmetry_ratio",
    "foreground_bbox_height_fraction", "foreground_bbox_width_fraction",
)
ACQUISITION_FEATURE_NAMES = (
    "content_height_fraction", "content_width_fraction", "pad_top_fraction",
    "pad_bottom_fraction", "pad_left_fraction", "pad_right_fraction", "source_aspect_ratio",
)
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PACKAGE_DIR.parent / "data/dxa_v1"


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_quality_source(source_dir):
    """Verify plain numeric source fits and their public/side preprocessing pins.

    Returns classifiers, model metadata and the original source manifest.
    source_dir may be checkpoints/source or checkpoints/source/quality.
    """
    source = Path(source_dir).resolve()
    quality = source if source.name == "quality" else source / "quality"
    spec = json.loads((quality / "manifest.json").read_text(encoding="utf-8"))
    filenames = {FEATURE_SPECS[task][2] for task in TASKS}
    side = spec.get("side_binding")
    if (spec.get("format") != SOURCE_FORMAT
            or spec.get("public_source_sha256") != PUBLIC_SOURCE_SHA
            or set(spec.get("files", {})) != filenames
            or set(spec.get("models", {})) != set(TASKS)
            or not isinstance(side, dict)
            or set(side) != {"backbone_sha256", "classifier_sha256"}
            or any(not _is_sha(value) for value in side.values())):
        raise ValueError("Invalid compact hip quality source manifest")
    values, metadata = {}, {}
    for task in TASKS:
        name = FEATURE_SPECS[task][2]
        path = (quality / name).resolve()
        if not path.is_relative_to(quality) or not path.is_file() or sha256(path) != spec["files"][name]:
            raise ValueError(f"Compact hip source checksum mismatch: {name}")
        values[task], metadata[task] = load_numeric_classifier(path, task, expected_metadata=spec["models"][task])
        if metadata[task].get("threshold") != values[task]["threshold"]:
            raise ValueError(f"Compact hip source threshold binding mismatch: {task}")
    if (set(metadata[TASKS[0]]["fit_image_ids"]) != set(metadata[TASKS[1]]["fit_image_ids"])
            or set(metadata[TASKS[0]]["fit_study_ids"]) != set(metadata[TASKS[1]]["fit_study_ids"])):
        raise ValueError("Both compact hip task fits must use the same labeled training set")
    return values, metadata, spec


def _verification_rows(data, count):
    """Select native real examples; this is export parity, not a quality score."""
    manifest, splits = data / "manifest.jsonl", data / "splits.json"
    if not manifest.is_file() or not splits.is_file():
        raise FileNotFoundError("Hip conversion requires prepared manifest.jsonl and splits.json")
    split = json.loads(splits.read_text(encoding="utf-8"))
    if split.get("manifest_sha256") != sha256(manifest):
        raise ValueError("Hip conversion data manifest/splits checksum mismatch")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted((row for row in rows if row.get("anatomical_region") in {"left_hip", "right_hip"}),
                  key=lambda row: row["image_id"])
    if len(rows) < 2 or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Hip conversion requires at least two unique real hip images")
    limit = min(len(rows), max(2, count))
    chosen = []
    for criterion in (
        lambda row: row.get("targets", {}).get("hip_roi") == 1,
        lambda row: row.get("targets", {}).get("hip_positioning_rotation") == 1,
        lambda row: row.get("anatomical_region") == "left_hip",
        lambda row: row.get("anatomical_region") == "right_hip",
    ):
        found = next((row for row in rows if criterion(row) and row not in chosen), None)
        if found is not None and len(chosen) < limit:
            chosen.append(found)
    chosen.extend(row for row in rows if row not in chosen)
    return chosen[:limit]


def _parity(graph, predictor, rows, data):
    import torch
    from combined_qc.router.features import load_record

    frames, canonical, geometry, sides = [], [], [], []
    for row in rows:
        native = load_record(row, data)  # verifies the original native pixel hash
        side = predictor.laterality_model.predict_array(native)
        if side["laterality"] not in {"left_hip", "right_hip"}:
            raise ValueError(f"Unavailable automatic side during conversion: {row['image_id']}")
        prepared, g = letterbox(native)
        if any(row.get(key) != value for key, value in g.items()):
            raise ValueError(f"Native acquisition geometry differs from prepared manifest: {row['image_id']}")
        frame = np.rint(prepared * 255).astype(np.uint8)
        frames.append(frame)
        canonical.append(frame[:, ::-1].copy() if side["laterality"] == "right_hip" else frame)
        geometry.append(g)
        sides.append(side["laterality"])
    frames, canonical = np.stack(frames), np.stack(canonical)
    feature_errors = {name: 0. for name, _ in OUTPUT_SPECS}
    probability_errors = {task: 0. for task in TASKS}
    decisions = {task: 0 for task in TASKS}
    numeric_errors = {task: 0. for task in TASKS}
    # Single images plus two-image batches check the actual exported batch axis.
    batches = [[index] for index in range(len(rows))] + [[0, 1]]
    for indices in batches:
        native_batch = np.ascontiguousarray(frames[indices])
        canonical_batch = np.ascontiguousarray(canonical[indices])
        with torch.inference_mode():
            native_source = {name: value.cpu().numpy() for (name, _), value in
                             zip(OUTPUT_SPECS, graph(torch.from_numpy(native_batch)))}
            canonical_source = {name: value.cpu().numpy() for (name, _), value in
                                zip(OUTPUT_SPECS, graph(torch.from_numpy(canonical_batch)))}
        native_onnx = predictor.extract_prepared(native_batch)
        canonical_onnx = predictor.extract_prepared(canonical_batch)
        for source, converted in ((native_source, native_onnx), (canonical_source, canonical_onnx)):
            for name, _ in OUTPUT_SPECS:
                error = float(np.max(np.abs(source[name] - converted[name])))
                feature_errors[name] = max(feature_errors[name], error)
                if not np.allclose(source[name], converted[name], atol=3e-4, rtol=3e-4):
                    raise ValueError(f"Shared hip ONNX feature parity failed: {name}")
        acquisition = np.stack([acquisition_features(geometry[index]) for index in indices])
        source_features = {
            "positioning_rotation": canonical_source["stage3_6400"],
            "roi": np.c_[native_source["global_512"], acquisition],
        }
        converted_features = {
            "positioning_rotation": canonical_onnx["stage3_6400"],
            "roi": np.c_[native_onnx["global_512"], acquisition],
        }
        for task in TASKS:
            values = predictor.classifiers[task]
            x = source_features[task].astype(np.float64)
            # Independent algebra checks the serialized scale/weights contract.
            logits = (x @ (values["coef"][0] / values["scale"])
                      + values["intercept"][0]
                      - np.dot(values["mean"] / values["scale"], values["coef"][0]))
            reference_numeric = np.exp(-np.logaddexp(0., -logits))
            reference = score_numeric(source_features[task], values)
            converted = score_numeric(converted_features[task], values)
            numeric_errors[task] = max(numeric_errors[task], float(np.max(np.abs(reference_numeric-reference))))
            probability_errors[task] = max(probability_errors[task], float(np.max(np.abs(reference-converted))))
            threshold = values["threshold"]
            decisions[task] += int(np.count_nonzero((reference >= threshold) != (converted >= threshold)))
    native_label_changes = 0
    for index in range(len(rows)):
        result = predictor.predict_prepared(frames[index].astype(np.float32) / 255.,
                                            geometry[index], sides[index])
        for task in TASKS:
            feature_index = "stage3_6400" if task == "positioning_rotation" else "global_512"
            view = canonical[index:index+1] if task == "positioning_rotation" else frames[index:index+1]
            with torch.inference_mode():
                values = graph(torch.from_numpy(np.ascontiguousarray(view)))
            position = [name for name, _ in OUTPUT_SPECS].index(feature_index)
            source_features = values[position].cpu().numpy()
            if task == "roi":
                source_features = np.c_[source_features, acquisition_features(geometry[index])[None]]
            source_probability = float(score_numeric(source_features, predictor.classifiers[task])[0])
            native_label_changes += int(result["tasks"][task]["violation"] !=
                                       (source_probability >= predictor.classifiers[task]["threshold"]))
    if (max(probability_errors.values()) > 3e-5
            or max(numeric_errors.values()) > 1e-10
            or sum(decisions.values()) or native_label_changes):
        raise ValueError("Compact hip conversion changed numeric probabilities or task labels")
    return {
        "real_image_ids": [row["image_id"] for row in rows], "real_samples": len(rows),
        "verified_native_pixel_hashes": True, "automatic_side_used": True,
        "batch_sizes_checked": [1, 2], "feature_max_absolute_errors": feature_errors,
        "numeric_only_probability_max_absolute_errors": numeric_errors,
        "probability_max_absolute_errors": probability_errors,
        "threshold_decision_mismatches": decisions, "native_inference_label_mismatches": native_label_changes,
        "probability_tolerance": 3e-5, "feature_tolerance": {"absolute": 3e-4, "relative": 3e-4},
    }


def convert_all(checkpoints_dir=None, *, data=None, device="auto", verbose=False, ci=False, verify_samples=2):
    """Regenerate one quality graph and copy two verified numeric final fits.

    No models are fitted here. Existing runtime remains intact if any source,
    checksum or real-image conversion verification fails. verify_samples counts
    real images in total; at least two are checked for dynamic-batch parity.
    """
    if device not in {"auto", "cpu", "mps", "cuda", "cuda:0"}:
        raise ValueError("Unsupported hip conversion device")
    if type(verify_samples) is not int or verify_samples < 1:
        raise ValueError("verify_samples must be a positive integer")
    enabled = bool(verbose or ci)
    with quiet_output(enabled):
        return _convert(checkpoints_dir, data=data, device=device, verbose=verbose, ci=ci,
                        verify_samples=verify_samples)


def _convert(checkpoints_dir, *, data, device, verbose, ci, verify_samples):
    import onnx
    import torch
    from .bootstrap import load_source_graph
    from .laterality.conversion import export_runtime

    started = time.perf_counter()
    enabled = bool(verbose or ci)
    progress = Progress("hip", enabled)
    base, source, runtime = resolve_checkpoint_paths("hip", checkpoints_dir)
    data = Path(data or DEFAULT_DATA).resolve()
    _, metadata, source_manifest = load_quality_source(source)
    graph, public = load_source_graph(base)
    graph = graph.cpu().eval().requires_grad_(False)
    if (public.get("source_sha256") != PUBLIC_SOURCE_SHA
            or public.get("feature_state_sha256") != FEATURE_STATE_SHA):
        raise ValueError("Hip public quality graph does not match its numeric source binding")
    rows = _verification_rows(data, verify_samples)
    preprocess = {
        "image_size": 320, "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "quality_input": "uint8 full 320px letterbox",
        "letterbox": "bilinear, round-half-up, centered black padding",
        "input": "native uint8 MONOCHROME2 or native grayscale PNG; no LUT/windowing",
        "positioning_mirror": "right_hip only", "roi_mirror": False,
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "acquisition_feature_names": list(ACQUISITION_FEATURE_NAMES),
        "quality_features": {task: {"feature_view": FEATURE_SPECS[task][0],
                                    "dimensions": FEATURE_SPECS[task][1]} for task in TASKS},
        "aggregate": "one classifier threshold per task; logical OR of quality labels",
    }
    inventory = {"quality_encoder": {"file": ENCODER_NAME, "source_weights_sha256": PUBLIC_SOURCE_SHA}}
    for task in TASKS:
        view, dimensions, filename = FEATURE_SPECS[task]
        inventory[task] = {"file": filename, "feature_view": view, "feature_dimension": dimensions,
                           "threshold": metadata[task]["threshold"]}
    base.mkdir(parents=True, exist_ok=True)
    progress.update("conversion", "Verified two final numeric fits and frozen public quality source", 0, 4)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    with tempfile.TemporaryDirectory(prefix=".compact-export-", dir=base) as temporary:
        stage = Path(temporary) / "runtime"
        stage.mkdir()
        try:
            with torch.inference_mode(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                torch.onnx.export(
                    graph, (torch.zeros((1, 320, 320), dtype=torch.uint8),), stage / ENCODER_NAME,
                    input_names=["gray_u8"], output_names=[name for name, _ in OUTPUT_SPECS],
                    dynamic_axes={name: {0: "batch"} for name in ["gray_u8", *[key for key, _ in OUTPUT_SPECS]]},
                    opset_version=17, dynamo=False, external_data=False,
                )
            onnx.checker.check_model(str(stage / ENCODER_NAME))
            for task in TASKS:
                filename = FEATURE_SPECS[task][2]
                shutil.copy2(source / "quality" / filename, stage / filename)
            shutil.copy2(source / "quality/manifest.json", stage / "quality_source.json")
            write_json(stage / "preprocess.json", preprocess)
            write_json(stage / "models.json", inventory)
            progress.update("conversion", "Exported one quality ResNet18 and copied two numeric LRs", 1, 4)
            side_report = export_runtime(source / "laterality", stage / "laterality", data=data,
                                         verbose=verbose, ci=ci, verify_samples=max(2, verify_samples))
            side_report["runtime_dir"] = str(runtime / "laterality")
            write_json(stage / "laterality/conversion_report.json", side_report)
            progress.update("conversion", "Exported the separate automatic side model pair", 2, 4)

            def seal():
                files = {str(path.relative_to(stage)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                         for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "manifest.json"}
                # Child manifest is required even though parent manifest cannot hash itself.
                files["laterality/manifest.json"] = {
                    "bytes": (stage / "laterality/manifest.json").stat().st_size,
                    "sha256": sha256(stage / "laterality/manifest.json")}
                write_json(stage / "manifest.json", {
                    "format": FORMAT, "format_version": 2, "files": files,
                    "public_encoder": public, "side_binding": source_manifest["side_binding"],
                    "quality_neural_models": 1, "quality_numeric_models": 2,
                    "auxiliary_neural_models": 1, "auxiliary_numeric_models": 1,
                    "source_quality_manifest_sha256": sha256(source / "quality/manifest.json"),
                })

            seal()
            restored = HipONNXPredictor(stage, device="cpu", verbose=verbose, ci=ci, checkpoints_dir=base)
            parity = _parity(graph, restored, rows, data)
            report = {
                "format": "hip_compact_onnx_numeric_conversion_v1",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "requested_device": device, "conversion_device": "cpu", "opset": 17,
                "quality_graph_count": 1, "quality_numeric_model_count": 2,
                "graph_count_including_laterality": 2, "numeric_count_including_laterality": 3,
                "source_sha256": {
                    "resnet18_imagenet.pth": PUBLIC_SOURCE_SHA,
                    "quality/manifest.json": sha256(source / "quality/manifest.json"),
                    **{f"quality/{name}": value for name, value in source_manifest["files"].items()},
                },
                "verification_manifest_sha256": sha256(data / "manifest.jsonl"),
                "verification_splits_sha256": sha256(data / "splits.json"),
                "quality_parity": parity, "laterality": side_report,
                "maximum_probability_error": max(
                    max(parity["probability_max_absolute_errors"].values()),
                    side_report["parity"]["end_to_end_max_probability_difference"]),
                "threshold_decision_mismatches": sum(parity["threshold_decision_mismatches"].values()),
                "dynamic_batch_verified": True, "refitted_during_conversion": False,
                "classifier_source_bytes_preserved": True,
                "elapsed_seconds": time.perf_counter() - started,
                "note": "Export parity checks serialization, not unseen model quality. Final task LRs fit all labeled training images.",
            }
            write_json(stage / "export_validation.json", report)
            seal()
            # Verify the complete sealed inventory once more before publication.
            from .preprocessing import verify_bundle
            verify_bundle(stage)
            progress.update("conversion", "Real native single/batch2 probabilities and labels agree", 3, 4)
            del restored
            publish_directory(stage, runtime)
            progress.update("conversion", "Published verified compact ONNX/numeric runtime", 4, 4)
            return {
                "checkpoints_dir": str(base), "source_dir": str(source), "runtime_dir": str(runtime),
                "manifest_path": str(runtime / "manifest.json"),
                "report_path": str(runtime / "export_validation.json"),
                "model_paths": [str(runtime / ENCODER_NAME),
                                *[str(runtime / FEATURE_SPECS[task][2]) for task in TASKS]],
                "laterality_model_paths": [str(runtime / "laterality" / name)
                                           for name in ("resnet18_features.onnx", "laterality_logreg.npz")],
                "validation": report,
            }
        finally:
            torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda", "cuda:0"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--verify-samples", type=int, default=2)
    args = parser.parse_args(argv)
    result = convert_all(args.checkpoints_dir, data=args.data, device=args.device,
                         verbose=args.verbose, ci=args.ci, verify_samples=args.verify_samples)
    if args.verbose or args.ci:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
