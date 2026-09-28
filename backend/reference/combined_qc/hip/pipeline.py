"""Central hip API with the same call signatures as the spine module."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..common import Progress, gather_sources, image_payload, resolve_checkpoint_paths, quiet_output
from .annotation import annotate_image
from .preprocessing import hip_geometry_features, acquisition_features, letterbox, load_native, validate_geometry
from .runtime import HipONNXPredictor

HIP_TASKS = ("hip_positioning_rotation", "hip_roi")
TASK_NAMES = {"hip_positioning_rotation": "positioning_rotation", "hip_roi": "roi"}
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent.parent
DATA_ROOT = MODULE_ROOT.parent / "data" / "dxa_v1"


def train(data=None, *, checkpoints_dir=None, device="auto", epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    """Fit two quality LRs plus automatic side; optionally publish conversion."""
    with quiet_output(enabled=bool(verbose or ci)):
        from .training import train as train_impl
        return train_impl(data, checkpoints_dir=checkpoints_dir, device=device, epochs=epochs,
                          verbose=verbose, ci=ci, resume=resume, convert=convert)


def load_checkpoints(checkpoints_dir=None, *, device="auto", verbose=False, ci=False):
    """Load one shared quality graph, two LRs and the automatic side pair."""
    base, _, runtime = resolve_checkpoint_paths("hip", checkpoints_dir)
    if not (runtime / "manifest.json").is_file():
        raise FileNotFoundError(f"Hip runtime checkpoints are missing in {runtime}; run hip conversion first")
    with quiet_output(enabled=bool(verbose or ci)):
        return HipONNXPredictor(runtime, device=device, verbose=verbose, ci=ci, checkpoints_dir=base)


@lru_cache(maxsize=1)
def _known_records():
    """Match exact dataset paths, never guess patient side from filenames."""
    manifest = DATA_ROOT / "manifest.jsonl"
    if not manifest.is_file():
        return {}
    records = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("anatomical_region") not in {"left_hip", "right_hip"}:
            continue
        for key in ("image_path", "native_path"):
            if row.get(key):
                records[str((DATA_ROOT / row[key]).resolve())] = row
        sources = set(row.get("source_paths", []))
        if row.get("canonical_source_path"):
            sources.add(row["canonical_source_path"])
        for name in sources:
            records[str((PROJECT_ROOT / "source_data" / name).resolve())] = row
    return records


def _prepared_and_native(path, row):
    native = load_native(path)
    is_prepared = row is not None and path == (DATA_ROOT / row["image_path"]).resolve()
    if not is_prepared:
        prepared, geometry = letterbox(native)
        return prepared, geometry, native, "native_input"
    geometry = validate_geometry(row)
    if native.shape != (320, 320):
        raise ValueError(f"Known prepared hip PNG must be 320x320: {path}")
    prepared = native.astype(np.float32) / 255.
    source = PROJECT_ROOT / "source_data" / row.get("canonical_source_path", "__missing__")
    if source.is_file():
        original = load_native(source)
        expected, original_geometry = letterbox(original)
        if original_geometry != geometry or not np.array_equal(expected, prepared):
            raise ValueError(f"Prepared PNG or original metadata no longer matches its native export: {path}")
        return prepared, geometry, original, "native_dataset_export"
    left, top = geometry["pad_left"], geometry["pad_top"]
    content = native[top:top + geometry["resized_height"], left:left + geometry["resized_width"]]
    reconstructed = Image.fromarray(content).resize((geometry["original_width"], geometry["original_height"]),
                                                    Image.Resampling.BILINEAR)
    return prepared, geometry, np.asarray(reconstructed), "reconstructed_from_prepared_input"


def _geometry(prepared, letterbox_geometry, native, laterality, config):
    g = letterbox_geometry
    h, w = native.shape
    def native_bbox(box):
        x0, y0, x1, y1 = box
        return [max(0., min(float(w), (x0-g["pad_left"])*w/g["resized_width"])),
                max(0., min(float(h), (y0-g["pad_top"])*h/g["resized_height"])),
                max(0., min(float(w), (x1-g["pad_left"])*w/g["resized_width"])),
                max(0., min(float(h), (y1-g["pad_top"])*h/g["resized_height"]))]
    foreground = prepared > max(.02, float(np.quantile(prepared, .75)) * .10)
    ys, xs = np.where(foreground)
    bbox = native_bbox([int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]) if len(ys) else None
    regions = []
    for name, bounds in (("whole", (.05, .05, .95, .95)),
                         ("superior_joint", (.15, .05, .85, .35)),
                         ("joint_field", (.08, .18, .92, .68)),
                         ("inferior_shaft", (.20, .65, .80, .98)),
                         ("lateral_left", (.02, .30, .40, .75)),
                         ("lateral_right", (.60, .30, .98, .75))):
        prepared_bbox = [round(value * 320) for value in bounds]
        regions.append({"name": name, "prepared_bbox": prepared_bbox,
                        "native_bbox": native_bbox(prepared_bbox), "type": "fixed_pixel_sampling_region"})
    positioning_pixels = np.fliplr(prepared).copy() if laterality == "right_hip" else prepared
    return {"coordinate_system": "native_pixels_x_right_y_down", "image_width": w, "image_height": h,
            "letterbox": g, "prepared_coordinate_system": "320px_letterbox_x_right_y_down",
            "foreground_bbox": bbox, "pixel_regions": regions,
            "feature_names": config["geometry_feature_names"],
            "roi_pixel_features": hip_geometry_features(prepared).tolist(),
            "positioning_pixel_features": hip_geometry_features(positioning_pixels).tolist(),
            "acquisition_feature_names": config["acquisition_feature_names"],
            "acquisition_features": acquisition_features(g).tolist(),
            "positioning_mirrored": laterality == "right_hip",
            "interpretation": "Fixed pixel-region and brightness proxies; no anatomical points or ROI segmentation"}


def infer(source, *, checkpoints_dir=None, device="auto", verbose=False, ci=False,
          model=None, recursive=False):
    """Return labels, task data, native geometry and a ready PNG per input.

    One file returns a dictionary; a directory returns a sorted list. Side is
    predicted from native pixels for every image. Known prepared PNGs retain
    original acquisition geometry and are checked against their native export.
    """
    paths, is_folder = gather_sources(source, recursive=recursive)
    records = _known_records()
    inputs = [(path, records.get(str(path))) for path in paths]
    if model is not None and not isinstance(model, HipONNXPredictor):
        raise ValueError("model must be a hip handle returned by load_checkpoints")
    if model is not None:
        HipONNXPredictor._providers(device)
        if str(device).lower() != "auto" and str(device).lower().split(":")[0] != model.device:
            raise ValueError("Provided model and requested device do not match")
    if model is not None and checkpoints_dir is not None:
        base, _, _ = resolve_checkpoint_paths("hip", checkpoints_dir)
        if model.checkpoints_dir != base:
            raise ValueError("Provided model and checkpoints_dir do not match")
    predictor = model if model is not None else load_checkpoints(checkpoints_dir, device=device, verbose=verbose, ci=ci)
    progress = Progress("hip", enabled=bool(verbose or ci))
    results = []
    for index, (path, row) in enumerate(inputs, start=1):
        progress.update("infer", f"Processing {path.name}", current=index - 1, total=len(inputs))
        prepared, original_geometry, native, visualization_source = _prepared_and_native(path, row)
        side_prediction = predictor.laterality_model.predict_array(native, source=path)
        side = side_prediction["laterality"]
        if side is None:
            task_data = {name: {"violation": None, "score": None,
                                "reason": side_prediction["metadata"].get("reason", "Unavailable side")}
                         for name in HIP_TASKS}
            result = {"region": "hip", "source": str(path), "status": "invalid_image",
                      "laterality": None, "laterality_prediction": side_prediction, "needs_review": True,
                      "labels": {name: None for name in HIP_TASKS},
                      "scores": {name: None for name in HIP_TASKS}, "tasks": task_data, "any_violation": None,
                      "metadata": {"laterality": None, "laterality_source": "model",
                                   "laterality_confidence": None, "reason": task_data[HIP_TASKS[0]]["reason"],
                                   "image_id": row.get("image_id") if row else None,
                                   "checkpoints_dir": str(predictor.checkpoints_dir)},
                      "geometry": {"coordinate_system": "native_pixels_x_right_y_down",
                                   "image_width": native.shape[1], "image_height": native.shape[0],
                                   "foreground_bbox": None, "letterbox": original_geometry},
                      "image": image_payload(Image.fromarray(native))}
            result["annotated_image"] = annotate_image(result)
            results.append(result)
            progress.update("infer", f"Unavailable: {path.name}", current=index, total=len(inputs))
            continue
        if side not in {"left_hip", "right_hip"}:
            raise ValueError("Automatic hip classifier returned an unknown side")
        raw = predictor.predict_prepared(prepared, original_geometry, side)
        result = {"region": "hip", "source": str(path),
                  "laterality": side, "laterality_prediction": side_prediction,
                  "needs_review": bool(side_prediction["needs_review"]),
                  "labels": {name: bool(raw["tasks"][raw_name]["violation"]) for name, raw_name in TASK_NAMES.items()},
                  "scores": {name: raw["tasks"][raw_name]["score"] for name, raw_name in TASK_NAMES.items()},
                  "tasks": {name: raw["tasks"][raw_name] for name, raw_name in TASK_NAMES.items()},
                  "any_violation": bool(raw["any_violation"]),
                  "metadata": {"laterality": side, "laterality_source": "model",
                               "laterality_confidence": side_prediction["confidence"],
                               "image_id": row.get("image_id") if row else None,
                               "study_id": row.get("study_id") if row else None,
                               "checkpoints_dir": str(predictor.checkpoints_dir), "device": predictor.device,
                               "score_type": "single_logreg_threshold_centered",
                               "models": predictor.provenance,
                               "visualization_source": visualization_source, "note": raw["note"]},
                  "geometry": _geometry(prepared, original_geometry, native, side, predictor.config),
                  "image": image_payload(Image.fromarray(native))}
        result["annotated_image"] = annotate_image(result)
        results.append(result)
        progress.update("infer", f"Finished {path.name}", current=index, total=len(inputs))
    return results if is_folder else results[0]
