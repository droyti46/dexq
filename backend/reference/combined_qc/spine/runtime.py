"""Spine inference from two ONNX backbones and one numeric logistic model.

The public ``predict_image`` and ``predict_records`` methods return the same
structured geometry and three criteria as the original spine pipeline. Neither
method opens PyTorch checkpoints or joblib files. Neural inference runs in ONNX
Runtime on CPU; pixel rules, morphology, heatmap decoding and composition stay
in Python. One logistic model is used for every image, so OOF mode is invalid.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .engine.axis import measure_axis
from .engine.composition import compose_spine_result
from .engine.coverage import measure_coverage
from .engine.morphology import _image_features
from .engine.utils import letterbox320, read_grayscale, safe_path, sha256
from .landmarks_portable import decode, preprocess
from .onnx_runtime import ONNXSpineModels
from .single_logreg import N_FEATURES, load_numeric_model
from ..common import Progress, resolve_checkpoint_paths
from .preprocessing import load_native


ROOT = Path(__file__).resolve().parent
METADATA_KEYS = ("image_id", "study_id", "legacy_view_id", "anatomical_region",
                 "eligible", "native_path", "native_sha256", "image_path", "image_sha256")


class PortableSpinePipeline:
    """Portable spine pipeline with one artifact logistic-regression model.

    ``classifier_dir`` contains ``artifact_logreg.npz``. Its numeric feature
    layout is checked against the native morphology extractor before scoring.
    """

    def __init__(self, root: str | Path | None = None,
                 onnx_dir: str | Path | None = None,
                 classifier_dir: str | Path | None = None,
                 binary_path: str | Path | None = None,
                 device: str = "cpu", *, checkpoints_dir=None,
                 verbose: bool = False, ci: bool = False):
        if device not in ("cpu", "auto"):
            raise ValueError("The one-model ONNX spine pipeline supports CPU only")
        self.root = Path(root).resolve() if root is not None else ROOT
        self.region = "spine"
        self.checkpoints_dir, _, runtime_dir = resolve_checkpoint_paths("spine", checkpoints_dir)
        if checkpoints_dir is None and root is not None:
            self.checkpoints_dir = self.root / "checkpoints"
            runtime_dir = self.checkpoints_dir / "runtime"
        progress = Progress("spine", verbose or ci)
        self.onnx_dir = (Path(onnx_dir).resolve() if onnx_dir is not None
                         else runtime_dir / "onnx")
        self.classifier_dir = (Path(classifier_dir).resolve() if classifier_dir is not None
                               else runtime_dir / "classifiers")
        self.binary_path = (Path(binary_path).resolve() if binary_path is not None
                            else self.classifier_dir / "artifact_logreg.npz")
        progress.update("load", "Checking artifact_logreg.npz", 1, 3)
        self.classifier = load_numeric_model(self.binary_path)
        model_manifest_path = self.classifier_dir / "model_manifest.json"
        if model_manifest_path.is_file():
            model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            if (model_manifest.get("format") != "spine_single_artifact_model_v1"
                    or model_manifest.get("model_file") != self.binary_path.name
                    or model_manifest.get("model_sha256") != sha256(self.binary_path)
                    or model_manifest.get("n_models_trained") != 1
                    or model_manifest.get("n_features") != N_FEATURES
                    or model_manifest.get("onnx_resnet_sha256")
                    != sha256(self.onnx_dir / "resnet18_features.onnx")):
                raise ValueError("Single spine classifier manifest does not match the model")
        else:
            raise ValueError("Missing manifest for the single spine classifier")
        self.morphology_names = self.classifier.morphology_names
        if len(self.morphology_names) != 120:
            raise ValueError("Expected 120 native morphology features")
        progress.update("load", "artifact_logreg.npz loaded", 1, 3)
        self.neural = ONNXSpineModels(self.onnx_dir, progress=progress)
        self.provenance = {
            "device": "cpu_onnxruntime",
            "onnx_manifest_sha256": sha256(self.onnx_dir / "export_manifest.json"),
            "classifier_binary_sha256": sha256(self.binary_path),
            "classifier_count": 1,
            "classifier_model_manifest_sha256": (
                sha256(model_manifest_path) if model_manifest_path.is_file() else None),
        }

    def _feature_banks(self, image_ids: list[str], native_arrays: list[np.ndarray],
                       prepared_arrays: list[np.ndarray], batch_size: int = 16) -> dict:
        if (not image_ids or len(image_ids) != len(native_arrays)
                or len(image_ids) != len(prepared_arrays)
                or any(a.dtype != np.uint8 or a.shape != (320, 320) for a in prepared_arrays)):
            raise ValueError("Expected aligned native and prepared 320×320 grayscale images")
        morphology = []
        for native in native_arrays:
            descriptors = _image_features(native)
            if tuple(descriptors) != self.morphology_names:
                raise ValueError("Morphology feature order differs from the saved classifier")
            morphology.append(list(descriptors.values()))
        morphology = np.asarray(morphology, dtype=np.float32)
        global_arrays, spatial_arrays = [], []
        for start in range(0, len(image_ids), batch_size):
            batch = np.stack(prepared_arrays[start:start + batch_size])
            global_512, spatial_2048 = self.neural.resnet_features(batch)
            global_arrays.append(global_512)
            spatial_arrays.append(spatial_2048)
        global_features = np.concatenate([np.concatenate(global_arrays), morphology], axis=1)
        spatial_features = np.concatenate(spatial_arrays)
        if (global_features.shape != (len(image_ids), 632)
                or spatial_features.shape != (len(image_ids), 2048)
                or not np.isfinite(global_features).all()
                or not np.isfinite(spatial_features).all()):
            raise ValueError("Invalid portable feature banks")
        return {"resnet18_morph": {"image_ids": np.asarray(image_ids),
                                   "global_features": global_features,
                                   "spatial_features": spatial_features}}

    def _classify(self, banks: dict, records: list[dict]) -> dict:
        image_ids = [row["image_id"] for row in records]
        bank = banks["resnet18_morph"]
        if bank["image_ids"].tolist() != image_ids:
            raise ValueError("Feature bank row order changed")
        features = np.concatenate([bank["global_features"], bank["spatial_features"]], axis=1)
        if features.shape != (len(records), N_FEATURES) or not np.isfinite(features).all():
            raise ValueError("Invalid classifier input features")
        probabilities = self.classifier.predict_proba(features)
        if probabilities.shape != (len(records),) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid single-classifier probabilities")
        return {image_id: {"p_spine_artifacts": float(score)}
                for image_id, score in zip(image_ids, probabilities)}

    def _predict_arrays(self, records: list[dict], native_arrays: list[np.ndarray],
                        prepared_arrays: list[np.ndarray]) -> list[dict]:
        started = time.perf_counter()
        image_ids = [row["image_id"] for row in records]
        banks = self._feature_banks(image_ids, native_arrays, prepared_arrays)
        scores = self._classify(banks, records)
        coverage_results = [measure_coverage(array) for array in native_arrays]
        predictions, landmarks = [], []
        for row, native, coverage in zip(records, native_arrays, coverage_results):
            tensor, transform = preprocess(native)
            maps = self.neural.spine_maps(tensor)
            detected = decode({key: value[0] for key, value in maps.items()}, transform,
                              threshold=5.0)
            axis = measure_axis(detected)
            image_id = row["image_id"]
            result = compose_spine_result(image_id, detected, coverage, axis, scores[image_id])
            predictions.append({"image_id": image_id,
                                "legacy_view_id": row.get("legacy_view_id", image_id),
                                "eligible_for_evaluation": False,
                                "classifier_fold": None, "classifier_variant": "single",
                                "result": result})
            landmarks.append({"image_id": image_id, "landmark_result": detected,
                              "coverage_measurement": coverage, "axis_measurement": axis,
                              "criterion_scores": scores[image_id]})
        self.last_feature_banks = banks
        self.last_landmarks = landmarks
        self.last_run = {"n_images": len(records), "device": "cpu_onnxruntime",
                         "elapsed_seconds": time.perf_counter() - started,
                         "axis_preprocessing": "direct_square_resize",
                         "artifact_classifiers_loaded": 1}
        return predictions

    def predict_records(self, records: list[dict], dataset_dir: str | Path,
                        mode: str = "full") -> list[dict]:
        """Return predictions in caller order using the same model for every image."""
        if mode == "oof":
            raise ValueError("OOF inference is unavailable with one artifact classifier; use mode='full'")
        if mode != "full":
            raise ValueError("mode must be 'full'")
        if not records:
            return []
        metadata = [{key: row[key] for key in METADATA_KEYS if key in row} for row in records]
        image_ids = [row["image_id"] for row in metadata]
        if (any(not isinstance(image_id, str) or not image_id for image_id in image_ids)
                or len(image_ids) != len(set(image_ids))):
            raise ValueError("Nonempty unique image IDs required")
        native_arrays, prepared_arrays = [], []
        for row in metadata:
            if row.get("anatomical_region", "lumbar_spine") != "lumbar_spine":
                raise ValueError("This pipeline expects caller-supplied lumbar spine images")
            for kind, arrays in (("native", native_arrays), ("image", prepared_arrays)):
                expected = row.get(kind + "_sha256")
                path = safe_path(dataset_dir, row[kind + "_path"], expected)
                arrays.append(read_grayscale(path))
        return self._predict_arrays(metadata, native_arrays, prepared_arrays)

    def predict_image(self, path: str | Path, *, include_image: bool = False) -> dict:
        """Infer a native PNG, optionally returning its exact input pixels."""
        path = Path(path).resolve()
        array = load_native(path)
        image_id = "input_" + sha256(path)[:20]
        rows = [{"image_id": image_id, "legacy_view_id": path.stem}]
        result = self._predict_arrays(rows, [array], [letterbox320(array)])[0]["result"]
        if include_image:
            import base64
            from io import BytesIO
            from PIL import Image

            buffer = BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            result["image"] = {
                "encoding": "base64_png", "mode": "L",
                "width": int(array.shape[1]), "height": int(array.shape[0]),
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        return result
