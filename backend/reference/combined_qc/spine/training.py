"""CI-ready training of one artifact classifier and frozen backbone preparation."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile

from ..common import Progress, resolve_checkpoint_paths, write_json
from .engine.utils import sha256

ROOT = Path(__file__).resolve().parent


def train(data=None, *, checkpoints_dir=None, device="auto", epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    from ..common import quiet_output
    with quiet_output(verbose or ci):
        return _train(data, checkpoints_dir=checkpoints_dir, device=device, epochs=epochs,
                      verbose=verbose, ci=ci, resume=resume, convert=convert)


def _train(data=None, *, checkpoints_dir=None, device="auto", epochs=None,
           verbose=False, ci=False, resume=True, convert=True):
    if device not in ("auto", "cpu"):
        raise ValueError("The frozen spine feature extractor and logistic fit use CPU")
    if epochs is not None and (type(epochs) is not int or epochs != 1):
        raise ValueError("Spine fits one logistic regression; use epochs=None or 1")
    data_dir = Path(data).resolve() if data is not None else ROOT / "data"
    base, source, runtime = resolve_checkpoint_paths("spine", checkpoints_dir)
    progress = Progress("spine", verbose or ci)
    from .bootstrap import ensure_sources
    ensure_sources(source, verbose=verbose, ci=ci)
    # Extractor conversion is needed before ML fitting. No classifiers are
    # required here, so this also works with an initially empty runtime folder.
    from .onnx_models import export_models
    from .onnx_runtime import ONNXSpineModels
    onnx_dir = runtime / "onnx"
    if not (onnx_dir / "export_manifest.json").is_file():
        progress.update("train.prepare", "Converting both frozen backbones")
        export_models(onnx_dir, root=ROOT, source_dir=source, verify_samples=2, data=data_dir)
    ONNXSpineModels(onnx_dir)
    onnx_spec = json.loads((onnx_dir / "export_manifest.json").read_text())
    source_meta = json.loads((source / "resnet18.json").read_text())
    if (onnx_spec["source_weights"]["resnet18"]["state_sha256"] != source_meta["state_sha256"]
            or onnx_spec["source_weights"]["spinenet"]["weights_sha256"] != sha256(source / "vertebra_landmark.pth")):
        raise ValueError("Training ONNX extractors differ from the pinned source checkpoints")
    from .train_single import _read_records, PARAMETERS
    images, indices, labels = _read_records(data_dir)
    expected_ids = {images[i]["image_id"] for i in indices}
    manifest_path = source / "classifiers/model_manifest.json"
    resumed = False
    if resume and manifest_path.is_file():
        spec = json.loads(manifest_path.read_text())
        from .single_logreg import load_numeric_model
        model_path = manifest_path.parent / "artifact_logreg.npz"
        if (spec.get("image_manifest_sha256") == sha256(data_dir / "manifest.jsonl")
                and spec.get("corrected_labels_sha256") == sha256(data_dir / "labeled_manifest.jsonl")
                and spec.get("onnx_resnet_sha256") == sha256(onnx_dir / "resnet18_features.onnx")
                and spec.get("classifier_parameters") == PARAMETERS
                and spec.get("n_models_trained") == 1
                and set(spec.get("train_image_ids", [])) == expected_ids
                and spec.get("model_sha256") == sha256(model_path)):
            load_numeric_model(model_path)
            resumed = True
            progress.update("train.resume", "Verified completed single-classifier fit", 1, 1)
    if not resumed:
        from .train_single import train_single
        source.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=source, prefix="fit-") as temp:
            staging = Path(temp) / "classifiers"
            spec = train_single(root=ROOT, data=data_dir, output_dir=staging,
                                onnx_dir=onnx_dir, progress=progress)
            from ..common import publish_directory
            publish_directory(staging, source / "classifiers")
    conversion = None
    if convert:
        from .convert_all import convert_all
        conversion = convert_all(checkpoints_dir=base, data=data_dir, device=device,
                                 verbose=verbose, ci=ci, verify_samples=2)
    report = {"region": "spine", "checkpoints_dir": str(base), "source_dir": str(source),
              "runtime_dir": str(runtime), "resumed": resumed, "trained_models": 0 if resumed else 1,
              "frozen_models": ["ResNet18", "SpineNet"], "classifier": spec,
              "converted": bool(convert), "conversion": conversion}
    write_json(base / "training_report.json", report)
    progress.update("train", "Completed; checkpoints saved", 1, 1)
    return report
