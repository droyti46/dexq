"""Portable hip inference: NumPy, Pillow and ONNX Runtime only."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np

from ..common import Progress
from .preprocessing import (
    acquisition_features, centered_score, hip_geometry_features, letterbox,
    load_native, validate_geometry, verify_bundle,
)


TASKS = ("positioning_rotation", "roi")


class HipONNXPredictor:
    """Verified V13 graphs with persistent sessions reused between images."""

    region = "hip"

    def __init__(self, bundle, providers=None, *, device="auto", verbose=False, ci=False,
                 checkpoints_dir=None):
        import onnxruntime as ort
        progress = Progress("hip", enabled=bool(verbose or ci))
        progress.update("load", "Checking checkpoint inventory and checksums")
        self.root, self.manifest = verify_bundle(bundle)
        self.runtime_dir = self.root
        self.checkpoints_dir = Path(checkpoints_dir).resolve() if checkpoints_dir is not None else self.root.parent
        self.config = json.loads((self.root / "preprocess.json").read_text(encoding="utf-8"))
        self.inventory = json.loads((self.root / "models.json").read_text(encoding="utf-8"))
        self.provenance = {"format": "onnx", "models": 11, "quality_models": 10,
                           "aggregate": "five_fold_threshold_centered_mean",
                           "independent_aggregate_validation": False}
        if self.config.get("image_size") != 320:
            raise ValueError("V13 requires 320x320 letterbox inputs")
        self.mean = np.asarray(self.config.get("image_mean"), dtype=np.float32)
        self.std = np.asarray(self.config.get("image_std"), dtype=np.float32)
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or not (self.std > 0).all()):
            raise ValueError("Invalid image normalization in the hip checkpoint")
        if (len(self.config.get("geometry_feature_names", [])) != 8
                or len(self.config.get("acquisition_feature_names", [])) != 7):
            raise ValueError("Hip metadata must define 8 pixel and 7 acquisition features")
        self.providers = providers or self._providers(device)
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        self.sessions = {}
        files = self.manifest["files"]
        side_files = {"laterality/manifest.json", "laterality/resnet18_features.onnx", "laterality/laterality_logreg.npz"}
        if not side_files.issubset(files):
            raise ValueError("Hip checkpoints lack the automatic side classifier; run hip train and conversion")
        for metadata in ("preprocess.json", "models.json"):
            if metadata not in files:
                raise ValueError(f"Hip manifest does not cover {metadata}")
        models = []
        for target in TASKS:
            inventory = self.inventory.get(target)
            if (not isinstance(inventory, list) or len(inventory) != 5
                    or any(not isinstance(item, dict) or type(item.get("fold")) is not int for item in inventory)
                    or {item.get("fold") for item in inventory if isinstance(item, dict)} != set(range(5))):
                raise ValueError(f"Hip {target} must contain exactly the five outer folds")
            for item in inventory:
                filename = item.get("file")
                threshold = item.get("threshold")
                if (not isinstance(filename, str) or filename not in files
                        or not (self.root / filename).resolve().is_relative_to(self.root)):
                    raise ValueError(f"Unverified hip model path: {filename}")
                if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                        or not np.isfinite(threshold) or not 0 < threshold < 1):
                    raise ValueError(f"Invalid threshold for hip {target} fold {item['fold']}")
                models.append((target, item))
        if len({item["file"] for _, item in models}) != 10:
            raise ValueError("Hip inventory must list ten distinct graphs")
        for index, (target, item) in enumerate(models, start=1):
            progress.update("load", f"Loading {target}, fold {item['fold']}", current=index - 1, total=len(models))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.log_severity_level = 3
            session = ort.InferenceSession(str(self.root / item["file"]), sess_options=options,
                                           providers=self.providers)
            expected = {"pixels": ("tensor(float)", [3, 320, 320]),
                        "geometry": ("tensor(float)", [8])}
            if target == "roi":
                expected["acquisition"] = ("tensor(double)", [7])
            actual = {node.name: (node.type, node.shape[1:]) for node in session.get_inputs()}
            if actual != expected or [node.name for node in session.get_outputs()] != ["probability"]:
                raise ValueError(f"Unexpected hip graph interface: {item['file']}")
            self.sessions[item["file"]] = session
            progress.update("load", f"Loaded {target}, fold {item['fold']}", current=index, total=len(models))
        from .laterality.runtime import LateralityClassifier
        self.laterality_model = LateralityClassifier(runtime_dir=self.root / "laterality",
                                                     verbose=verbose, ci=ci)
        self.provenance["laterality"] = self.laterality_model.provenance

    @staticmethod
    def _providers(device):
        import onnxruntime as ort
        device = str(device).lower()
        if device in {"auto", "cpu"}:
            return ["CPUExecutionProvider"]
        if device in {"cuda", "cuda:0"}:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise ValueError("CUDA inference requires onnxruntime-gpu and an available CUDA provider")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        raise ValueError("ONNX hip inference supports device='auto', 'cpu' or 'cuda'")

    def _run(self, filename, inputs):
        probability = np.asarray(self.sessions[filename].run(["probability"], inputs)[0])
        if probability.size != 1 or not np.isfinite(probability).all():
            raise ValueError(f"Hip model returned a nonfinite or malformed probability: {filename}")
        result = float(probability.ravel()[0])
        if not 0 <= result <= 1:
            raise ValueError(f"Hip model probability is outside [0,1]: {filename}")
        return result

    def predict_prepared(self, pixels, geometry, laterality, folds=None):
        """Score a 320px letterbox with verified, original-image metadata.

        Validation happens before feature extraction or model execution. This
        rejects NaN, zero dimensions, broken padding and inconsistent resizing.
        """
        if laterality not in {"left_hip", "right_hip"}:
            raise ValueError("Expected the validated left_hip/right_hip classifier result")
        pixels = np.asarray(pixels)
        if (pixels.shape != (320, 320) or not np.issubdtype(pixels.dtype, np.floating)
                or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1):
            raise ValueError("Prepared pixels must be float grayscale 320x320 in [0,1]")
        geometry = validate_geometry(geometry, self.config["image_size"])
        top, bottom = geometry["pad_top"], geometry["pad_bottom"]
        left, right = geometry["pad_left"], geometry["pad_right"]
        if ((top and np.any(pixels[:top] != 0)) or (bottom and np.any(pixels[-bottom:] != 0))
                or (left and np.any(pixels[:, :left] != 0)) or (right and np.any(pixels[:, -right:] != 0))):
            raise ValueError("Prepared pixels do not match the declared black letterbox padding")
        selected_folds = list(range(5)) if folds is None else folds
        try:
            selected_folds = list(selected_folds)
        except TypeError as exc:
            raise ValueError("folds must be a nonempty subset of integer folds 0..4") from exc
        if (not selected_folds or any(isinstance(fold, (bool, np.bool_))
                or not isinstance(fold, Integral) or not 0 <= fold < 5 for fold in selected_folds)):
            raise ValueError("folds must be a nonempty subset of integer folds 0..4")
        selected = set(selected_folds)
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]

        def inputs(array):
            features = hip_geometry_features(array)
            if features.shape != (8,) or not np.isfinite(features).all():
                raise ValueError("Hip pixel features must be eight finite values")
            return {"pixels": np.ascontiguousarray((np.repeat(array[None, None], 3, axis=1)-mean)/std,
                                                     dtype=np.float32),
                    "geometry": features[None].astype(np.float32)}

        p_inputs = inputs(np.fliplr(pixels).copy() if laterality == "right_hip" else pixels)
        roi_inputs = inputs(pixels)
        acquisition = acquisition_features(geometry)
        if acquisition.shape != (7,) or not np.isfinite(acquisition).all():
            raise ValueError("Hip acquisition features must be seven finite values")
        roi_inputs["acquisition"] = acquisition[None].astype(np.float64)
        result = {"laterality": laterality, "tasks": {}}
        for target, model_inputs in (("positioning_rotation", p_inputs), ("roi", roi_inputs)):
            details = []
            for item in self.inventory[target]:
                if item["fold"] not in selected:
                    continue
                probability = self._run(item["file"], model_inputs)
                threshold = float(item["threshold"])
                score = float(centered_score(probability, threshold))
                if not np.isfinite(score):
                    raise ValueError(f"Nonfinite centered score for {target} fold {item['fold']}")
                details.append({"fold": item["fold"], "probability": probability, "threshold": threshold,
                                "violation": int(probability >= threshold), "centered_score": score})
            score = float(np.mean([item["centered_score"] for item in details]))
            result["tasks"][target] = {"score": score, "threshold": .5,
                                       "violation": int(score >= .5), "folds": details}
        result["any_violation"] = int(any(value["violation"] for value in result["tasks"].values()))
        result["note"] = ("1=quality violation, not a diagnosis. Cross-fold aggregate is a deployment rule "
                          "without independent validation; its score is not a calibrated disease probability.")
        return result

    def predict(self, image):
        native = load_native(image)
        side = self.laterality_model.predict_array(native, source=image)
        if side["laterality"] is None:
            raise ValueError(side["metadata"].get("reason", "Automatic side classification unavailable"))
        pixels, geometry = letterbox(native, self.config["image_size"])
        result = self.predict_prepared(pixels, geometry, side["laterality"])
        result["laterality_prediction"] = side
        result["needs_review"] = side["needs_review"]
        return result
