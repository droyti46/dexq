"""Portable compact hip inference: one quality ONNX and two numeric LRs.

The automatic side encoder/LR remains a separate auxiliary pair. Inference
never trains, downloads weights or imports a training framework.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import Progress
from .preprocessing import (
    acquisition_features, centered_score, letterbox, load_native,
    validate_geometry, verify_bundle,
)

TASKS = ("positioning_rotation", "roi")
FORMAT = "hip_compact_resnet18_numeric_lr_runtime_v1"
SOURCE_FORMAT = "hip_compact_quality_source_v1"
ENCODER_NAME = "resnet18_quality_features.onnx"
PUBLIC_SOURCE_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
FEATURE_STATE_SHA = "6125408e356323c4193a8ab1f56bb155aad05af3fc2b542e3c5a8587e9a17505"
FEATURE_SPECS = {
    "positioning_rotation": ("stage3_6400", 6400, "positioning_logreg.npz"),
    "roi": ("global512_acquisition7", 519, "roi_logreg.npz"),
}
OUTPUT_SPECS = (("global_512", 512), ("spatial_2048", 2048), ("stage3_6400", 6400))


def load_numeric_classifier(path, task, *, expected_metadata=None):
    """Validate a plain numeric checkpoint; serialized Python objects are banned."""
    if task not in FEATURE_SPECS:
        raise ValueError("Unknown hip quality task")
    feature_view, dimensions, _ = FEATURE_SPECS[task]
    fields = {"mean", "scale", "coef", "intercept", "threshold", "classes", "metadata_json"}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != fields:
            raise ValueError(f"Unexpected numeric hip checkpoint fields: {task}")
        arrays = {key: np.asarray(saved[key]).copy()
                  for key in ("mean", "scale", "coef", "intercept", "threshold")}
        classes = np.asarray(saved["classes"])
        raw_metadata = np.asarray(saved["metadata_json"])
        if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
            raise ValueError("Hip classifier metadata must be scalar Unicode JSON")
        metadata = json.loads(raw_metadata.item())
    if (any(arrays[key].shape != (dimensions,) for key in ("mean", "scale"))
            or arrays["coef"].shape != (1, dimensions)
            or arrays["intercept"].shape != (1,) or arrays["threshold"].shape != ()
            or any(value.dtype != np.float64 or not np.isfinite(value).all()
                   for value in arrays.values())
            or np.any(arrays["scale"] <= 0)
            or not 0 < float(arrays["threshold"]) < 1
            or classes.dtype != np.int64 or classes.shape != (2,)
            or classes.tolist() != [0, 1]):
        raise ValueError(f"Invalid numeric hip classifier arrays: {task}")
    if (not isinstance(metadata, dict) or metadata.get("task") != task
            or metadata.get("feature_view") != feature_view
            or metadata.get("feature_dimension") != dimensions
            or metadata.get("source_weights_sha256") != PUBLIC_SOURCE_SHA
            or (expected_metadata is not None and metadata != expected_metadata)):
        raise ValueError(f"Hip classifier feature/source binding mismatch: {task}")
    for key in ("fit_image_ids", "fit_study_ids"):
        identifiers = metadata.get(key)
        if (not isinstance(identifiers, list) or not identifiers
                or any(not isinstance(item, str) or not item for item in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise ValueError(f"Invalid hip classifier {key}: {task}")
    arrays["threshold"] = float(arrays["threshold"])
    return arrays, metadata


def score_numeric(features, values):
    """FP64 logistic inference with explicit overflow and finite-value checks."""
    features = np.asarray(features)
    dimensions = len(values["mean"])
    if (features.ndim != 2 or not len(features) or features.shape[1] != dimensions
            or features.dtype.kind not in {"f", "i", "u"}
            or not np.isfinite(features).all()):
        raise ValueError(f"Expected finite numeric [N,{dimensions}] hip features")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            logits = ((features.astype(np.float64) - values["mean"]) / values["scale"]) @ values["coef"].T
            logits = logits[:, 0] + values["intercept"][0]
    except FloatingPointError as error:
        raise ValueError("Hip classifier numeric normalization overflow") from error
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite hip classifier logits")
    # This form remains finite even for very large positive/negative logits.
    probabilities = np.exp(-np.logaddexp(0., -logits))
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite hip quality probabilities")
    return probabilities


class HipONNXPredictor:
    """Verified compact quality models with persistent ONNX sessions."""

    region = "hip"

    def __init__(self, bundle, providers=None, *, device="auto", verbose=False, ci=False,
                 checkpoints_dir=None):
        import onnxruntime as ort

        progress = Progress("hip", enabled=bool(verbose or ci))
        progress.update("load", "Checking compact checkpoint inventory and checksums")
        self.root, self.manifest = verify_bundle(bundle)
        self.runtime_dir = self.root
        self.checkpoints_dir = Path(checkpoints_dir).resolve() if checkpoints_dir is not None else self.root.parent
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Hip runtime requires the compact format; run current hip training and conversion")
        files = self.manifest["files"]
        required = {ENCODER_NAME, "positioning_logreg.npz", "roi_logreg.npz",
                    "preprocess.json", "models.json", "quality_source.json",
                    "laterality/manifest.json", "laterality/resnet18_features.onnx",
                    "laterality/laterality_logreg.npz"}
        if not required.issubset(files):
            raise ValueError("Hip manifest does not cover all quality and automatic-side models")
        public = self.manifest.get("public_encoder", {})
        if (public.get("source_sha256") != PUBLIC_SOURCE_SHA
                or public.get("feature_state_sha256") != FEATURE_STATE_SHA
                or public.get("fine_tuned_on_dxa") is not False):
            raise ValueError("Hip quality encoder differs from the pinned frozen ImageNet source")
        self.config = json.loads((self.root / "preprocess.json").read_text(encoding="utf-8"))
        self.inventory = json.loads((self.root / "models.json").read_text(encoding="utf-8"))
        source = json.loads((self.root / "quality_source.json").read_text(encoding="utf-8"))
        if (source.get("format") != SOURCE_FORMAT
                or source.get("public_source_sha256") != PUBLIC_SOURCE_SHA
                or set(source.get("files", {})) != {"positioning_logreg.npz", "roi_logreg.npz"}
                or set(source.get("models", {})) != set(TASKS)):
            raise ValueError("Invalid copied hip quality source manifest")
        if (self.config.get("image_size") != 320
                or self.config.get("positioning_mirror") != "right_hip only"
                or self.config.get("roi_mirror") is not False
                or self.config.get("quality_input") != "uint8 full 320px letterbox"):
            raise ValueError("Unexpected compact hip preprocessing")
        if (self.config.get("image_mean") != [0.485, 0.456, 0.406]
                or self.config.get("image_std") != [0.229, 0.224, 0.225]
                or len(self.config.get("geometry_feature_names", [])) != 8
                or len(self.config.get("acquisition_feature_names", [])) != 7):
            raise ValueError("Invalid compact hip normalization or feature metadata")
        if (set(self.inventory) != {"quality_encoder", *TASKS}
                or self.inventory["quality_encoder"].get("file") != ENCODER_NAME
                or self.inventory["quality_encoder"].get("source_weights_sha256") != PUBLIC_SOURCE_SHA):
            raise ValueError("Unexpected compact hip model inventory")
        self.classifiers, self.classifier_metadata = {}, {}
        for task in TASKS:
            view, dimensions, filename = FEATURE_SPECS[task]
            item = self.inventory[task]
            if (not isinstance(item, dict) or item.get("file") != filename
                    or item.get("feature_view") != view or item.get("feature_dimension") != dimensions
                    or source["files"][filename] != files[filename]["sha256"]):
                raise ValueError(f"Invalid compact hip classifier inventory: {task}")
            model, metadata = load_numeric_classifier(self.root / filename, task,
                                                      expected_metadata=source["models"][task])
            if item.get("threshold") != model["threshold"]:
                raise ValueError(f"Hip classifier threshold binding mismatch: {task}")
            self.classifiers[task], self.classifier_metadata[task] = model, metadata
            progress.update("load", f"Loaded numeric classifier: {task}", len(self.classifiers), 4)
        self.providers = list(providers) if providers is not None else self._providers(device)
        if not self.providers or any(not isinstance(provider, str) for provider in self.providers):
            raise ValueError("providers must be a nonempty list of ONNX provider names")
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.log_severity_level = 3
        self.encoder = ort.InferenceSession(str(self.root / ENCODER_NAME), sess_options=options,
                                           providers=self.providers)
        inputs, outputs = self.encoder.get_inputs(), self.encoder.get_outputs()
        if ([(node.name, node.type) for node in inputs] != [("gray_u8", "tensor(uint8)")]
                or inputs[0].shape[1:] != [320, 320]
                or [(node.name, node.type, node.shape[1:]) for node in outputs]
                != [(name, "tensor(float)", [size]) for name, size in OUTPUT_SPECS]):
            raise ValueError("Unexpected shared hip quality ONNX interface")
        self.sessions = {ENCODER_NAME: self.encoder}
        progress.update("load", "Loaded one shared frozen ResNet18 quality encoder", 3, 4)
        from .laterality.runtime import LateralityClassifier
        self.laterality_model = LateralityClassifier(runtime_dir=self.root / "laterality",
                                                     verbose=verbose, ci=ci)
        actual_side = {key: self.laterality_model.provenance[key]
                       for key in ("backbone_sha256", "classifier_sha256")}
        if (source.get("side_binding") != actual_side
                or self.manifest.get("side_binding") != actual_side):
            raise ValueError("Hip automatic-side pair differs from quality training preprocessing")
        self.provenance = {
            "format": "onnx+numeric_npz", "models": 5,
            "quality_neural_models": 1, "quality_numeric_models": 2,
            "quality_models": 3, "auxiliary_neural_models": 1, "auxiliary_numeric_models": 1,
            "aggregate": "single_classifier_threshold_per_task_then_or",
            "quality_encoder_sha256": files[ENCODER_NAME]["sha256"],
            "public_source_sha256": PUBLIC_SOURCE_SHA,
            "n_fit_images": {task: len(meta["fit_image_ids"]) for task, meta in self.classifier_metadata.items()},
            "independent_final_checkpoint_validation": False,
            "laterality": self.laterality_model.provenance,
        }
        progress.update("load", "Loaded and bound automatic side classifier", 4, 4)

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

    def extract_prepared(self, frames, *, batch_size=16):
        """Return verified shared features from prepared uint8 letterboxes."""
        frames = np.asarray(frames)
        if (frames.dtype != np.uint8 or frames.ndim != 3 or not len(frames)
                or frames.shape[1:] != (320, 320)
                or type(batch_size) is not int or batch_size <= 0):
            raise ValueError("Expected nonempty uint8 [N,320,320] frames and positive batch_size")
        parts = [[] for _ in OUTPUT_SPECS]
        for start in range(0, len(frames), batch_size):
            batch = np.ascontiguousarray(frames[start:start + batch_size])
            result = self.encoder.run(None, {"gray_u8": batch})
            if len(result) != len(OUTPUT_SPECS):
                raise ValueError("Malformed shared hip feature output count")
            for values, target, (_, dimensions) in zip(result, parts, OUTPUT_SPECS):
                if values.shape != (len(batch), dimensions) or not np.isfinite(values).all():
                    raise ValueError("Malformed or nonfinite shared hip feature output")
                target.append(values)
        return {name: np.vstack(parts[index]) for index, (name, _) in enumerate(OUTPUT_SPECS)}

    def predict_prepared(self, pixels, geometry, laterality, folds=None):
        """Score an original-image letterbox after validating pixels/geometry.

        Production uses one final classifier per task. Fold fits belong only
        to the separate validation script and cannot be selected here.
        """
        if folds is not None:
            raise ValueError("Compact production inference has no folds; use the validation script")
        if laterality not in {"left_hip", "right_hip"}:
            raise ValueError("Expected the validated left_hip/right_hip classifier result")
        pixels = np.asarray(pixels)
        if (pixels.shape != (320, 320) or not np.issubdtype(pixels.dtype, np.floating)
                or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1):
            raise ValueError("Prepared pixels must be finite float grayscale 320x320 in [0,1]")
        geometry = validate_geometry(geometry, self.config["image_size"])
        top, bottom = geometry["pad_top"], geometry["pad_bottom"]
        left, right = geometry["pad_left"], geometry["pad_right"]
        if ((top and np.any(pixels[:top] != 0)) or (bottom and np.any(pixels[-bottom:] != 0))
                or (left and np.any(pixels[:, :left] != 0)) or (right and np.any(pixels[:, -right:] != 0))):
            raise ValueError("Prepared pixels do not match the declared black letterbox padding")
        frame = np.rint(pixels.astype(np.float64) * 255).astype(np.uint8)
        # All training/native preprocessing uses a uint8 letterbox; accepting
        # arbitrary continuous values here would silently change that contract.
        if not np.allclose(pixels, frame.astype(np.float64) / 255., atol=1e-7, rtol=0.):
            raise ValueError("Prepared pixels must be the normalized uint8 letterbox produced by preprocessing")
        canonical = np.fliplr(frame).copy() if laterality == "right_hip" else frame
        same_view = np.array_equal(frame, canonical)
        frames = frame[None] if same_view else np.stack((frame, canonical))
        features = self.extract_prepared(frames)
        p_index = 0 if same_view else 1
        quality_features = {
            "positioning_rotation": features["stage3_6400"][p_index:p_index + 1],
            "roi": np.r_[features["global_512"][0], acquisition_features(geometry)][None],
        }
        result = {"laterality": laterality, "tasks": {}, "quality_forward_views": len(frames)}
        for task in TASKS:
            model = self.classifiers[task]
            probability = float(score_numeric(quality_features[task], model)[0])
            threshold = model["threshold"]
            score = float(centered_score(probability, threshold))
            if not np.isfinite(score):
                raise ValueError(f"Nonfinite centered hip score: {task}")
            result["tasks"][task] = {
                "score": score, "threshold": .5, "violation": int(probability >= threshold),
                "probability": probability, "probability_threshold": threshold,
                "model": FEATURE_SPECS[task][2],
                "score_type": "single_logreg_threshold_centered",
                "feature_view": FEATURE_SPECS[task][0],
            }
        result["any_violation"] = int(any(value["violation"] for value in result["tasks"].values()))
        result["note"] = ("1=quality violation. One frozen quality ResNet18 feeds two logistic classifiers. "
                          "Scores are uncalibrated quality scores; final classifiers are fitted on all "
                          "labeled training images. Independent estimates are produced by separate grouped validation.")
        return result

    def predict(self, image):
        native = load_native(image)
        side = self.laterality_model.predict_array(native, source=image)
        if side["laterality"] is None:
            raise ValueError(side["metadata"].get("reason", "Automatic side classification unavailable"))
        pixels, geometry = letterbox(native, self.config["image_size"])
        result = self.predict_prepared(pixels, geometry, side["laterality"])
        result["laterality_prediction"], result["needs_review"] = side, side["needs_review"]
        return result
