"""Convert local V13 training checkpoints into a verified, ONNX-only bundle.

Conversion is explicit: importing the module never changes the supplied bundle.
All eleven graphs and the numeric side classifier are staged and checked
before replacing checkpoints/runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _quiet(enabled):
    with warnings.catch_warnings():
        if not enabled:
            warnings.simplefilter("ignore")
            # ONNX Runtime may write its initial telemetry message through C++
            # stderr. Capture native descriptors as well as Python streams.
            sys.stdout.flush()
            sys.stderr.flush()
            saved = []
            with tempfile.TemporaryFile() as native_output:
                try:
                    for descriptor in (1, 2):
                        saved.append((descriptor, os.dup(descriptor)))
                        os.dup2(native_output.fileno(), descriptor)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        yield
                finally:
                    for descriptor, original in saved:
                        os.dup2(original, descriptor)
                        os.close(original)
        else:
            yield


def _verification_rows(data, verify_samples):
    """Select real held-out images, prioritizing ROI positives and hip sides."""
    manifest_path, splits_path = data / "manifest.jsonl", data / "splits.json"
    for path in (manifest_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Hip ONNX verification requires prepared data: {path}")
    split = json.loads(splits_path.read_text(encoding="utf-8"))
    if split.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Hip conversion data has mismatched manifest/splits checksums")
    fold_specs = split["folds"]
    fold_items = fold_specs.items() if isinstance(fold_specs, dict) else (
        (row.get("fold", index), row) for index, row in enumerate(fold_specs)
    )
    fold_by_id = {}
    for fold, spec in fold_items:
        for image_id in spec.get("val", spec.get("val_image_ids", [])):
            if image_id in fold_by_id:
                raise ValueError(f"Duplicate outer-fold assignment: {image_id}")
            fold_by_id[image_id] = int(fold)
    records = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["anatomical_region"] not in {"left_hip", "right_hip"} or not row["eligible"] or row["source_split"] != "train":
            continue
        if row["image_id"] not in fold_by_id:
            raise ValueError(f"Hip image lacks an outer fold: {row['image_id']}")
        row["fold"] = fold_by_id[row["image_id"]]
        records.append(row)
    if len(records) != 150 or len({row["image_id"] for row in records}) != 150:
        raise ValueError("Hip conversion expects the 150 unique prepared V13 training images")
    selected = {}
    for fold in range(5):
        candidates = sorted((row for row in records if row["fold"] == fold), key=lambda row: row["image_id"])
        if len(candidates) < 2:
            raise ValueError(f"Fold {fold} lacks two real images for dynamic-batch verification")
        # Prioritize a positive ROI example, then hip sides; fill deterministically.
        rows = []
        for choose in (
            lambda row: row["targets"]["hip_roi"] == 1,
            lambda row: row["anatomical_region"] == "left_hip",
            lambda row: row["anatomical_region"] == "right_hip",
        ):
            row = next((row for row in candidates if choose(row) and row not in rows), None)
            if row is not None and len(rows) < max(2, verify_samples):
                rows.append(row)
        rows += [row for row in candidates if row not in rows][:max(2, verify_samples) - len(rows)]
        selected[fold] = rows
    return selected


def _inputs(rows, data, preprocess, *, canonicalize):
    from .preprocessing import acquisition_features, hip_geometry_features

    images, geometries, acquisition = [], [], []
    for row in rows:
        with Image.open(data / row["image_path"]) as image:
            if image.mode != "L" or image.size != (320, 320):
                raise ValueError(f"Verification input is not a prepared 320x320 grayscale image: {row['image_id']}")
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        if canonicalize and row["anatomical_region"] == "right_hip":
            pixels = np.fliplr(pixels).copy()
        images.append(np.repeat(pixels[None], 3, axis=0))
        geometries.append(hip_geometry_features(pixels))
        acquisition.append(acquisition_features(row))
    mean = np.asarray(preprocess["image_mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(preprocess["image_std"], dtype=np.float32)[None, :, None, None]
    return {
        "pixels": np.ascontiguousarray((np.asarray(images, np.float32) - mean) / std),
        "geometry": np.ascontiguousarray(geometries, dtype=np.float32),
        "acquisition": np.ascontiguousarray(acquisition, dtype=np.float64),
    }


def _export_and_verify(graph, inputs, output, threshold, image_ids, *, roi, enabled):
    with _quiet(enabled):
        import onnx
        import onnxruntime as ort
        import torch
        from .onnx_models import numpy_roi_reference

    names = ["pixels", "geometry"] + (["acquisition"] if roi else [])
    tensors = tuple(torch.from_numpy(inputs[name]) for name in names)
    graph.eval()
    with _quiet(enabled), torch.inference_mode():
        torch.onnx.export(
            graph, tuple(tensor[:1] for tensor in tensors), str(output),
            input_names=names, output_names=["probability"],
            dynamic_axes={name: {0: "batch"} for name in names + ["probability"]},
            opset_version=17, dynamo=False, external_data=False,
        )
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    errors, numeric_errors, mismatches = [], [], 0
    batches = [[index] for index in range(len(image_ids))] + [[0, 1]]
    for indices in batches:
        args = tuple(tensor[indices] for tensor in tensors)
        with torch.inference_mode():
            source = graph(*args).numpy()
        reference = numpy_roi_reference(graph, args) if roi else source
        converted = session.run(["probability"], {name: inputs[name][indices] for name in names})[0]
        if converted.shape != (len(indices), 1) or not np.isfinite(converted).all():
            raise ValueError(f"Invalid ONNX probability output: {output.name}")
        error = float(np.max(np.abs(converted - reference)))
        errors.append(error)
        numeric_errors.append(float(np.max(np.abs(source - reference))))
        mismatches += int(np.count_nonzero((converted >= threshold) != (reference >= threshold)))
    if max(errors) > 2e-5 or max(numeric_errors) > 1e-10 or mismatches:
        raise RuntimeError(f"Hip ONNX parity failed: {output.name}, max error={max(errors)}, label changes={mismatches}")
    del session
    return {
        "file": str(output.name), "real_image_ids": image_ids,
        "real_samples": len(image_ids), "batch_sizes_checked": [1, 2],
        "probability_max_abs_error": max(errors),
        "calibration_numeric_max_abs_error": max(numeric_errors),
        "threshold_decision_mismatches": mismatches, "tolerance": 2e-5,
        "opset": 17, "neural_dtype": "float32", "calibration_dtype": "float64" if roi else None,
    }


def _commit_runtime(stage, runtime):
    """Transactional directory replacement with rollback; source stays unchanged."""
    if runtime.is_symlink():
        raise ValueError("Refusing to replace a symlinked hip runtime directory")
    if runtime.exists() and not runtime.is_dir():
        raise ValueError("Hip runtime destination must be a directory")
    backup = runtime.with_name(f".runtime-backup-{uuid.uuid4().hex}")
    had_runtime = runtime.exists()
    if had_runtime:
        os.replace(runtime, backup)
    try:
        os.replace(stage, runtime)
    except BaseException:
        if had_runtime:
            os.replace(backup, runtime)
        raise
    if had_runtime:
        shutil.rmtree(backup)


def convert_all(checkpoints_dir=None, *, data=None, device="auto", verbose=False, ci=False, verify_samples=2):
    """Export all hip models offline and verify each on real images and batch=2.

    ``verify_samples`` is the number of held-out images per outer fold (minimum
    two for the dynamic batch check). DL uses FP32, calibration uses FP64.
    CPU export is deliberate: FP64 calibration is unsupported on Apple's MPS.
    An existing runtime is preserved if any validation or export fails.
    """
    from combined_qc.common import Progress, resolve_checkpoint_paths

    if device not in {"auto", "cpu", "mps", "cuda", "cuda:0"}:
        raise ValueError("Unsupported hip conversion device")
    if isinstance(verify_samples, bool) or not isinstance(verify_samples, int) or verify_samples < 1:
        raise ValueError("verify_samples must be a positive integer")
    enabled = bool(verbose or ci)
    progress = Progress("hip", enabled=enabled)
    base, source, runtime = resolve_checkpoint_paths("hip", checkpoints_dir)
    data = Path(data or DEFAULT_DATA).resolve()
    required = [source / f"fold_{fold}_{suffix}" for fold in range(5) for suffix in (
        "positioning_v13.pt", "roi_v13_ensemble.pt", "roi_v13_calibration.json",
    )]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained hip source checkpoint: {path}")
    config_dir = source / "pretrained/resnet18"
    if not (config_dir / "config.json").is_file():
        config_dir = PACKAGE_DIR / "checkpoints/source/pretrained/resnet18"
    if not (config_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing offline architecture config: {config_dir / 'config.json'}")
    processor = json.loads((config_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    preprocess = {
        "image_size": 320, "dropout": 0.35,
        "image_mean": processor["image_mean"], "image_std": processor["image_std"],
        "letterbox": "bilinear, round-half-up, centered black padding",
        "input": "native uint8 MONOCHROME2 or native grayscale PNG; no LUT/windowing",
        "positioning_mirror": "right_hip only", "roi_mirror": False,
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "acquisition_feature_names": list(ACQUISITION_FEATURE_NAMES),
        "aggregate": "mean(sigmoid(logit(p_fold)-logit(threshold_fold))) >= 0.5",
    }
    means, scales = np.asarray(preprocess["image_mean"]), np.asarray(preprocess["image_std"])
    if means.shape != (3,) or scales.shape != (3,) or not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid local hip image normalization config")
    verification = _verification_rows(data, verify_samples)
    progress.update("conversion", "Validated source checkpoints and real verification images", current=0, total=11)
    # Heavy training dependencies are imported only by an explicit conversion call.
    with _quiet(enabled):
        import torch
        from .onnx_models import PositioningGraph, ROIGraph, checked_threshold, model_from_state
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".runtime-export-", dir=base))
    try:
        (stage / "models").mkdir()
        (stage / "calibration").mkdir()
        (stage / "hf_config").mkdir()
        shutil.copy2(config_dir / "config.json", stage / "hf_config/config.json")
        _json(stage / "preprocess.json", preprocess)
        inventory = {"positioning_rotation": [], "roi": []}
        checks = []
        for fold in range(5):
            rows = verification[fold]
            ids = [row["image_id"] for row in rows]
            checkpoint = torch.load(source / f"fold_{fold}_positioning_v13.pt", map_location="cpu", weights_only=True)
            if checkpoint.get("fold") != fold or checkpoint.get("target") != "positioning_rotation" or checkpoint.get("canonicalize_laterality") is not True:
                raise ValueError(f"Unsupported positioning checkpoint protocol in fold {fold}")
            threshold = checked_threshold(checkpoint["threshold"])
            graph = PositioningGraph(model_from_state(checkpoint["state_dict"], config_dir)).eval()
            inputs = _inputs(rows, data, preprocess, canonicalize=True)
            filename = f"models/positioning_fold_{fold}.onnx"
            checks.append(_export_and_verify(graph, inputs, stage / filename, threshold, ids, roi=False, enabled=enabled))
            inventory["positioning_rotation"].append({"fold": fold, "file": filename, "threshold": threshold})
            progress.update("conversion", f"Positioning fold {fold}: ONNX and real-image parity passed", current=2 * fold + 1, total=11)
            del graph, checkpoint, inputs
            gc.collect()

            checkpoint = torch.load(source / f"fold_{fold}_roi_v13_ensemble.pt", map_location="cpu", weights_only=True)
            calibration = json.loads((source / f"fold_{fold}_roi_v13_calibration.json").read_text(encoding="utf-8"))
            if checkpoint.get("fold") != fold or checkpoint.get("target") != "roi" or checkpoint.get("inference") != "native":
                raise ValueError(f"Unsupported ROI checkpoint protocol in fold {fold}")
            if calibration.get("fold") != fold or calibration.get("feature_order") != ["centered_neural_logit", "acquisition_metadata_logit"] or calibration.get("acquisition_feature_names") != list(ACQUISITION_FEATURE_NAMES):
                raise ValueError(f"Unsupported ROI calibration feature ordering in fold {fold}")
            members = checkpoint["members"]
            if len(members) != 4 or {member["inner_fold"] for member in members} != set(range(5)) - {fold}:
                raise ValueError(f"ROI fold {fold} must contain its four distinct inner members")
            threshold = checked_threshold(calibration["threshold"])
            graph = ROIGraph(
                [model_from_state(member["state_dict"], config_dir) for member in members],
                [member["threshold"] for member in members], calibration,
            ).eval()
            inputs = _inputs(rows, data, preprocess, canonicalize=False)
            filename = f"models/roi_fold_{fold}.onnx"
            checks.append(_export_and_verify(graph, inputs, stage / filename, threshold, ids, roi=True, enabled=enabled))
            inventory["roi"].append({"fold": fold, "file": filename, "threshold": threshold, "neural_members": 4})
            _json(stage / f"calibration/roi_fold_{fold}.json", calibration)
            progress.update("conversion", f"ROI fold {fold}: four members, FP64 calibration and parity passed", current=2 * fold + 2, total=11)
            del graph, checkpoint, calibration, members, inputs
            gc.collect()
        from .laterality.conversion import export_runtime
        progress.update("conversion", "Exporting automatic left-right classifier and frozen ResNet18")
        side_report = export_runtime(source / "laterality", stage / "laterality", data=data,
                                     verbose=verbose, ci=ci, verify_samples=verify_samples)
        side_report["runtime_dir"] = str(runtime / "laterality")
        _json(stage / "laterality/conversion_report.json", side_report)
        _json(stage / "models.json", inventory)
        report = {
            "format": "hip_v13_onnx_conversion_v1", "exported_at": datetime.now(timezone.utc).isoformat(),
            "requested_device": device, "conversion_device": "cpu", "opset": 17,
            "source_sha256": {path.name: _sha256(path) for path in required},
            "architecture_config_sha256": _sha256(config_dir / "config.json"),
            "normalization_config_sha256": _sha256(config_dir / "preprocessor_config.json"),
            "verification_manifest_sha256": _sha256(data / "manifest.jsonl"),
            "verification_splits_sha256": _sha256(data / "splits.json"),
            "graphs": checks, "graph_count": 11, "quality_graph_count": 10,
            "laterality": side_report,
            "maximum_probability_error": max(
                max(check["probability_max_abs_error"] for check in checks),
                side_report["parity"]["end_to_end_max_probability_difference"],
            ),
            "threshold_decision_mismatches": sum(check["threshold_decision_mismatches"] for check in checks),
            "dynamic_batch_verified": True,
            "note": "Conversion parity is not an estimate of model quality; each exported graph reproduces its source fold.",
        }
        _json(stage / "export_validation.json", report)
        files = {
            str(path.relative_to(stage)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        _json(stage / "manifest.json", {"format_version": 1, "files": files})
        _commit_runtime(stage, runtime)
        progress.update("conversion", "Committed verified ONNX runtime bundle", current=11, total=11)
        return {"checkpoints_dir": str(base), "source_dir": str(source), "runtime_dir": str(runtime),
                "manifest_path": str(runtime / "manifest.json"), "report_path": str(runtime / "export_validation.json"),
                "model_paths": [str(runtime / item["file"]) for task in inventory.values() for item in task],
                "laterality_model_paths": [str(runtime / "laterality" / name)
                                           for name in ("resnet18_features.onnx", "laterality_logreg.npz")],
                "validation": report}
    finally:
        torch.set_num_threads(previous_threads)
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda", "cuda:0"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--verify-samples", type=int, default=2)
    args = parser.parse_args(argv)
    convert_all(args.checkpoints_dir, data=args.data, device=args.device, verbose=args.verbose,
                ci=args.ci, verify_samples=args.verify_samples)


if __name__ == "__main__":
    main()
