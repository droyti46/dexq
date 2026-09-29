"""Convert saved checkpoints only: two ONNX graphs and one numeric ML model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from ..common import Progress, publish_directory, resolve_checkpoint_paths, write_json
from .engine.utils import sha256

ROOT = Path(__file__).resolve().parent


def convert_all(checkpoints_dir=None, *, data=None, device="auto", verbose=False,
                ci=False, verify_samples=2):
    """Export from source into a staged runtime; never train inside conversion."""
    from ..common import quiet_output
    with quiet_output(verbose or ci):
        return _convert_all(checkpoints_dir, data=data, device=device, verbose=verbose,
                            ci=ci, verify_samples=verify_samples)


def _convert_all(checkpoints_dir=None, *, data=None, device="auto", verbose=False,
                 ci=False, verify_samples=2):
    if device not in {"auto", "cpu"}:
        raise ValueError("Spine conversion uses CPU")
    if not isinstance(verify_samples, int) or verify_samples < 1:
        raise ValueError("verify_samples must be a positive integer")
    base, source, runtime = resolve_checkpoint_paths("spine", checkpoints_dir)
    from .bootstrap import ensure_sources
    ensure_sources(source, verbose=verbose, ci=ci)
    model_path = source / "classifiers/artifact_logreg.npz"
    manifest_path = source / "classifiers/model_manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Train first: source/classifiers must contain the saved ML checkpoint")
    spec = json.loads(manifest_path.read_text())
    if sha256(model_path) != spec.get("model_sha256"):
        raise ValueError("Source ML checkpoint checksum mismatch")
    from .bootstrap import RESNET_STATE_SHA
    # Older saved manifests can be bound by the exact known original graph.
    original_graph_sha = "9506741d36b44aae108e6015f6da92d22df89570ed692b395024129a133cdc0a"
    if (spec.get("resnet_state_sha256") != RESNET_STATE_SHA
            and not (spec.get("resnet_state_sha256") is None
                     and spec.get("onnx_resnet_sha256") == original_graph_sha)):
        raise ValueError("ML checkpoint was fitted with a different ResNet18 extractor; retrain first")
    progress = Progress("spine", verbose or ci)
    from .onnx_models import export_models
    from .single_logreg import load_numeric_model
    load_numeric_model(model_path)
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix="convert-") as temp:
        stage = Path(temp) / "runtime"
        stage.mkdir()
        progress.update("convert", "Exporting ResNet18 and SpineNet", 0, 3)
        onnx_report = export_models(stage / "onnx", root=ROOT, source_dir=source,
                                    verify_samples=verify_samples, data=data)
        progress.update("convert", "Both ONNX graphs passed PyTorch parity", 2, 3)
        shutil.copytree(source / "classifiers", stage / "classifiers")
        # NPZ already stores numeric coefficients. Conversion copies them
        # unchanged and associates them with the verified ONNX extractor.
        spec["onnx_resnet_sha256"] = sha256(stage / "onnx/resnet18_features.onnx")
        spec["resnet_state_sha256"] = RESNET_STATE_SHA
        write_json(stage / "classifiers/model_manifest.json", spec)
        load_numeric_model(stage / "classifiers/artifact_logreg.npz")
        progress.update("convert", "artifact_logreg.npz verified (no fitting)", 3, 3)
        from .runtime import PortableSpinePipeline
        PortableSpinePipeline(onnx_dir=stage / "onnx", classifier_dir=stage / "classifiers")
        report = {"format": "spine_saved_checkpoint_conversion_v2",
                  "region": "spine", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "checkpoints_dir": str(base), "neural_models": onnx_report,
                  "artifact_classifier": {"format": "numeric_npz", "n_models": 1,
                    "sha256": sha256(model_path), "training_performed": False},
                  "source_artifacts": {str(p.relative_to(source)): sha256(p) for p in
                    [source / "resnet18.pt", source / "resnet18.json",
                     source / "vertebra_landmark.pth", model_path, manifest_path]},
                  "training_performed": False}
        write_json(stage / "conversion_report.json", report)
        publish_directory(stage, runtime)
    progress.update("convert", "Published validated runtime", 3, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-samples", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = convert_all(args.checkpoints_dir, data=args.data, device=args.device,
                         verify_samples=args.verify_samples, verbose=args.verbose, ci=args.ci)
    if args.output:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
