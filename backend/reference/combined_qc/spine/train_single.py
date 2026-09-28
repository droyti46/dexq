"""Fit the one production spine-artifact logistic model on all 99 labels.

The frozen ONNX ResNet18 supplies 512 global and 2048 spatial descriptors;
120 native-pixel descriptors are inserted between them. Exactly one sklearn
Pipeline.fit call is made. The fitted numerical parameters are serialized into
a non-pickle NPZ for ONNX/NumPy inference. No folds or OOF scores are produced.

Run from the repository root::

    python -m combined_qc.spine.train_single
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .engine.morphology import _image_features
from .engine.utils import read_grayscale, read_manifest, safe_path, sha256
from .onnx_runtime import ONNXSpineModels
from .single_logreg import N_FEATURES, load_numeric_model, save_numeric_model


ROOT = Path(__file__).resolve().parent
PARAMETERS = {"C": 0.01, "class_weight": "balanced", "max_iter": 2000,
              "random_state": 20260920, "solver": "liblinear", "tol": 1e-5}


def _read_records(data_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    images = read_manifest(data_dir / "manifest.jsonl")
    labeled = read_manifest(data_dir / "labeled_manifest.jsonl")
    if len(images) != 100 or len(labeled) != 100:
        raise ValueError("Expected exactly 100 spine images and label records")
    by_id = {row["image_id"]: row for row in labeled}
    if (len(by_id) != 100 or len({row["image_id"] for row in images}) != 100
            or len({row["study_id"] for row in images}) != 100
            or set(by_id) != {row["image_id"] for row in images}):
        raise ValueError("Image/label IDs or study membership disagree")
    targets = []
    for image in images:
        label = by_id[image["image_id"]]
        target = label["targets"]["spine_artifacts"]
        if (image["study_id"] != label["study_id"]
                or target not in (0, 1, None)
                or (target is not None and type(target) is not int)):
            raise ValueError("Invalid corrected artifact label")
        targets.append(target)
    train_indices = np.asarray([i for i, target in enumerate(targets)
                                if target is not None], dtype=int)
    labels = np.asarray([targets[i] for i in train_indices], dtype=int)
    if len(train_indices) != 99 or int(labels.sum()) != 17:
        raise ValueError("Expected 99 labeled images with 17 artifact positives")
    return images, train_indices, labels


def _extract_features(images: list[dict], data_dir: Path, onnx_dir: Path,
                      batch_size: int = 16, progress=None) -> tuple[np.ndarray, list[str]]:
    neural = ONNXSpineModels(onnx_dir)
    prepared, morphology = [], []
    morphology_names = None
    for index, image in enumerate(images, 1):
        native = read_grayscale(safe_path(data_dir, image["native_path"],
                                          image["native_sha256"]))
        frame = read_grayscale(safe_path(data_dir, image["image_path"],
                                         image["image_sha256"]))
        if frame.dtype != np.uint8 or frame.shape != (320, 320):
            raise ValueError("Expected prepared uint8 320×320 ResNet18 image")
        descriptors = _image_features(native)
        if morphology_names is None:
            morphology_names = list(descriptors)
            if len(morphology_names) != 120:
                raise ValueError("Expected 120 native morphology features")
        elif list(descriptors) != morphology_names:
            raise ValueError("Native morphology feature order changed")
        prepared.append(frame)
        morphology.append(list(descriptors.values()))
        if progress is not None:
            progress.update("train.prepare", image["image_id"], index, len(images))
    global_parts, spatial_parts = [], []
    for start in range(0, len(images), batch_size):
        global_512, spatial_2048 = neural.resnet_features(
            np.stack(prepared[start:start + batch_size]))
        global_parts.append(global_512)
        spatial_parts.append(spatial_2048)
        if progress is not None:
            progress.update("train.features", "Frozen ResNet18", min(start + batch_size, len(images)), len(images))
    features = np.ascontiguousarray(np.concatenate((
        np.concatenate(global_parts),
        np.asarray(morphology, dtype=np.float32),
        np.concatenate(spatial_parts),
    ), axis=1), dtype=np.float32)
    if features.shape != (100, N_FEATURES) or not np.isfinite(features).all():
        raise ValueError("Invalid 2680-dimensional artifact feature matrix")
    return features, morphology_names


def train_single(*, root: str | Path = ROOT, output_dir: str | Path | None = None,
                 onnx_dir: str | Path | None = None, data=None, progress=None) -> dict:
    root = Path(root).resolve()
    data_dir = Path(data).resolve() if data is not None else root / "data"
    output_dir = (Path(output_dir).resolve() if output_dir is not None
                  else root / "checkpoints/source/classifiers")
    onnx_dir = Path(onnx_dir).resolve() if onnx_dir is not None else root / "checkpoints/runtime/onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    images, train_indices, labels = _read_records(data_dir)
    extract_started = time.perf_counter()
    features, morphology_names = _extract_features(images, data_dir, onnx_dir, progress=progress)
    extract_seconds = time.perf_counter() - extract_started

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**PARAMETERS)),
    ])
    started = time.perf_counter()
    if progress is not None:
        progress.update("train.fit", "LogisticRegression: 99 labels / 2680 features", 0, 1)
    model.fit(features[train_indices], labels)
    if progress is not None:
        progress.update("train.fit", "LogisticRegression fitted", 1, 1)
    train_seconds = time.perf_counter() - started
    model_path = output_dir / "artifact_logreg.npz"
    save_numeric_model(model_path, model, train_prevalence=float(labels.mean()),
                       morphology_names=morphology_names)
    restored = load_numeric_model(model_path)
    source_raw = np.asarray(model.decision_function(features), dtype=np.float64)
    converted_raw = restored.decision_function(features)
    max_raw_difference = float(np.max(np.abs(source_raw - converted_raw)))
    if max_raw_difference > 1e-6:
        raise AssertionError("Single logistic NPZ differs from fitted classifier")
    report = {
        "format": "spine_single_artifact_model_v1",
        "model_file": model_path.name,
        "model_sha256": sha256(model_path),
        "n_models_trained": 1,
        "n_train": len(train_indices),
        "n_positive": int(labels.sum()),
        "n_unlabeled": len(images) - len(train_indices),
        "train_image_ids": [images[i]["image_id"] for i in train_indices],
        "feature_parts": ["resnet_global_512", "native_morphology_120",
                          "resnet_spatial_2048"],
        "n_features": N_FEATURES,
        "classifier_parameters": PARAMETERS,
        "train_prevalence": float(labels.mean()),
        "prior_strength": 0.5,
        "threshold": 0.5,
        "onnx_resnet_sha256": sha256(onnx_dir / "resnet18_features.onnx"),
        "resnet_state_sha256": json.loads((onnx_dir / "export_manifest.json").read_text())[
            "source_weights"]["resnet18"]["state_sha256"],
        "image_manifest_sha256": sha256(data_dir / "manifest.jsonl"),
        "corrected_labels_sha256": sha256(data_dir / "labeled_manifest.jsonl"),
        "feature_extraction_seconds": extract_seconds,
        "training_seconds": train_seconds,
        "npz_max_decision_function_difference": max_raw_difference,
        "evaluation": ("No independent test remains for this 99-case fit; "
                       "do not report its training-image metrics as held-out performance."),
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if progress is not None:
        progress.update("train.save", str(model_path), 1, 1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()
    from ..common import Progress
    report = train_single(root=args.root, output_dir=args.output_dir, onnx_dir=args.onnx_dir,
                          progress=Progress("spine", args.verbose or args.ci))



if __name__ == "__main__":
    main()
