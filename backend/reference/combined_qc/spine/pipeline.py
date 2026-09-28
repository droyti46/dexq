"""Public inference entry point for the standalone lumbar-spine module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .engine.utils import read_manifest

if TYPE_CHECKING:
    from .portable_inference import PortableSpinePipeline


SPINE_TASKS = {
    "spine_positioning": "coverage",
    "spine_axis": "axis",
    "spine_artifacts": "artifacts",
}


def train(data=None, *, checkpoints_dir=None, device="auto", epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    """Train the single artifact classifier; keep both DL backbones frozen."""
    from .training import train as train_impl
    return train_impl(data, checkpoints_dir=checkpoints_dir, device=device, epochs=epochs,
                      verbose=verbose, ci=ci, resume=resume, convert=convert)


def load_checkpoints(checkpoints_dir=None, *, device="auto", verbose=False, ci=False):
    """Validate and load both ONNX backbones and the one numeric classifier."""
    from ..common import quiet_output
    with quiet_output(verbose or ci):
        from .runtime import PortableSpinePipeline
        return PortableSpinePipeline(checkpoints_dir=checkpoints_dir, device=device,
                                     verbose=verbose, ci=ci)


def infer(source, *, checkpoints_dir=None, device="auto", verbose=False, ci=False,
          model=None, recursive=False):
    """Infer one native image or every PNG/DICOM in a directory.

    One file returns a dict; a directory returns a list in sorted path order.
    Ready PNG/base64 annotations are always returned, irrespective of labels.
    The model handle can be reused without reloading sessions.
    """
    from ..common import Progress, gather_sources, resolve_checkpoint_paths
    paths, is_folder = gather_sources(source, recursive=recursive)
    if model is None:
        model = load_checkpoints(checkpoints_dir, device=device, verbose=verbose, ci=ci)
    else:
        from .runtime import PortableSpinePipeline
        if not isinstance(model, PortableSpinePipeline):
            raise TypeError("Expected a spine checkpoint handle")
        if device not in ("auto", "cpu"):
            raise ValueError("The spine ONNX runtime supports CPU only")
        if checkpoints_dir is not None and model.checkpoints_dir != resolve_checkpoint_paths("spine", checkpoints_dir)[0]:
            raise ValueError("Model handle and checkpoints_dir disagree")
    progress = Progress("spine", verbose or ci)
    outputs = []
    for index, path in enumerate(paths, 1):
        outputs.append(_infer_one(path, model))
        progress.update("infer", path.name, index, len(paths))
    return outputs if is_folder else outputs[0]


def _infer_one(image_path, model):
    result = model.predict_image(image_path, include_image=True)
    image = result.pop("image")
    landmarks = result["shared_landmarks"]
    axis = result["axis"]
    gap_sequence = result["coverage"]["components"]["upper_gap_sequence"]
    top, bottom = axis.get("top_xy"), axis.get("bottom_xy")
    line = None if top is None or bottom is None else {
        "top_xy": top,
        "bottom_xy": bottom,
        "angle_deg": axis.get("angle_deg"),
        "signed_angle_deg": axis.get("signed_angle_deg"),
    }
    output = {
        "region": "spine",
        "source": str(Path(image_path).resolve()),
        "metadata": {"image_id": result["image_id"], "provenance": model.provenance,
                     "coordinate_system": "native_image_pixels"},
        "image_id": result["image_id"],
        "anatomical_region": "lumbar_spine",
        "labels": {
            "spine_positioning": result["coverage"]["violation"],
            "spine_axis": axis["violation"],
            "spine_artifacts": result["artifacts"]["violation"],
        },
        "scores": {
            "spine_positioning": result["coverage"]["score"],
            "spine_axis": axis["score"],
            "spine_artifacts": result["artifacts"]["score"],
        },
        "geometry": {
            "coordinate_system": landmarks["coordinate_system"],
            "image_width": landmarks["image_width"],
            "image_height": landmarks["image_height"],
            "axis_line": line,
            "vertebral_candidates": landmarks["vertebral_candidates"],
            "gap_lines": [
                {
                    "center_xy": [gap["x"], gap["y"]],
                    "endpoints_xy": gap["endpoints"],
                    "contrast": gap["contrast"],
                }
                for gap in gap_sequence["observed_gaps"]
            ],
            "pelvic_proxies": gap_sequence["pelvic_proxies"],
        },
        "criteria": {
            "spine_positioning": result["coverage"],
            "spine_axis": axis,
            "spine_artifacts": result["artifacts"],
        },
        "raw": result,
    }
    from .annotation import annotate_image
    output["annotated_image"] = annotate_image({"image": image, "geometry": output["geometry"]})
    output["tasks"] = output["criteria"]
    output["any_violation"] = result["quality"]["violation"]
    return output


def run_spine(root: str | Path, classifier_dir: str | Path | None = None,
              *, device: str = "auto") -> list[dict]:
    """Return inference rows for all known images using the one saved model.

    The rows include labels for analysis but are not OOF predictions. Use the
    model's saved training membership to choose any evaluation subset.
    """
    if device not in ("auto", "cpu"):
        raise ValueError("The ONNX spine pipeline currently supports CPU only")
    from .portable_inference import PortableSpinePipeline

    root = Path(root).resolve()
    records = read_manifest(root / "data/manifest.jsonl")
    labeled = {row["image_id"]: row for row in read_manifest(root / "data/labeled_manifest.jsonl")}
    model = PortableSpinePipeline(root=root, classifier_dir=classifier_dir)
    predictions = model.predict_records(records, root / "data", mode="full")
    if len(predictions) != len(records):
        raise ValueError("Spine pipeline did not return every image")
    rows = []
    for record, prediction in zip(records, predictions):
        if record["image_id"] != prediction["image_id"]:
            raise ValueError("Spine prediction order changed")
        truth = labeled[record["image_id"]]
        result = prediction["result"]
        row = {
            "image_id": record["image_id"],
            "study_id": record["study_id"],
            "legacy_view_id": record["legacy_view_id"],
            "region": "lumbar_spine",
            "fold": None,
            "source": "single_model_spine_inference",
            "out_of_fold": False,
        }
        for target, module in SPINE_TASKS.items():
            row[target + "_true"] = truth["targets"][target]
            row[target + "_prediction"] = result[module]["violation"]
            row[target + "_score"] = result[module].get("score")
        decisions = [row[target + "_prediction"] for target in SPINE_TASKS]
        row["spine_or_true"] = truth["quality_label"]
        row["spine_or_prediction"] = (True if any(v is True for v in decisions)
                                       else None if any(v is None for v in decisions) else False)
        rows.append(row)
    return rows
