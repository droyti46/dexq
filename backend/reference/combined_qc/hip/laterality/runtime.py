"""One ResNet18 ONNX and one safe numeric left/right classifier."""
from pathlib import Path
import json

import numpy as np

from combined_qc.common import Progress
from .features import DEFAULT_RUNTIME, HIP_ROOT, PREPROCESSING, REPRESENTATIONS, FeatureExtractor, prepare_native, sha256, subset


class LateralityClassifier:
    def __init__(self, checkpoints_dir=None, *, runtime_dir=None, verbose=False, ci=False):
        self.checkpoints_dir = Path(checkpoints_dir).resolve() if checkpoints_dir else HIP_ROOT / "checkpoints"
        runtime = Path(runtime_dir).resolve() if runtime_dir is not None else self.checkpoints_dir / "runtime/laterality"
        if runtime_dir is not None and checkpoints_dir is not None and runtime != self.checkpoints_dir / 'runtime/laterality':
            raise ValueError('Laterality runtime_dir and HIP checkpoints_dir disagree')
        self.runtime_dir = runtime
        manifest = json.loads((runtime / "manifest.json").read_text())
        if manifest.get("format") != "hip_laterality_runtime_v1" or set(manifest.get("files", {})) != {"resnet18_features.onnx", "laterality_logreg.npz"}:
            raise ValueError("Invalid laterality runtime inventory")
        for filename, digest in manifest["files"].items():
            path=(runtime/filename).resolve()
            if not path.is_relative_to(runtime) or sha256(path) != digest:
                raise ValueError(f"Laterality checkpoint checksum mismatch: {filename}")
        with np.load(runtime / "laterality_logreg.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"mean", "scale", "coef", "intercept", "classes", "metadata"}:
                raise ValueError("Invalid numeric checkpoint fields")
            self.mean, self.scale, self.coef, self.intercept = [np.asarray(saved[name], dtype=np.float64) for name in ("mean", "scale", "coef", "intercept")]
            self.metadata = json.loads(str(saved["metadata"].item()))
            classes = saved["classes"].tolist()
        spec = self.metadata
        n = REPRESENTATIONS.get(spec.get("representation"))
        if (spec != manifest.get("model") or spec.get("format") != "image_only_laterality_logreg_v1"
                or spec.get("class_names") != ["left_hip", "right_hip"] or classes != [0, 1]
                or n is None or spec.get("n_features") != n
                or any(array.shape != (n,) for array in (self.mean, self.scale, self.coef))
                or self.intercept.shape != (1,)
                or any(not np.isfinite(array).all() for array in (self.mean, self.scale, self.coef, self.intercept))
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid numeric laterality model or metadata")
        if (spec.get("preprocessing") != PREPROCESSING or spec.get("threshold") != .5
                or spec.get("review_threshold") != .9 or spec.get("mirror_average") is not False
                or spec.get("backbone_sha256") != manifest["files"]["resnet18_features.onnx"]):
            raise ValueError("Laterality preprocessing or backbone binding mismatch")
        progress = Progress("hip_laterality", verbose or ci)
        progress.update("load", "One numeric left/right classifier verified", 1, 2)
        self.extractor = FeatureExtractor(runtime / "resnet18_features.onnx",expected_sha=manifest['files']['resnet18_features.onnx'])
        self.provenance = {"classifier_sha256": manifest["files"]["laterality_logreg.npz"],
                           "backbone_sha256": manifest["files"]["resnet18_features.onnx"],
                           "n_fit_images": len(spec["train_image_ids"]),
                           "trained_on_holdout": spec["trained_on_holdout"],
                           "representation": spec["representation"]}
        progress.update("load", "Orientation-preserving ResNet18 ONNX loaded", 2, 2)

    def score_features(self, features):
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != 2560 or not len(features) or features.dtype.kind not in {"f", "i", "u"} or not np.isfinite(features).all():
            raise ValueError("Expected finite [N,2560] ResNet18 embeddings")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                z = ((subset(features, self.metadata["representation"]) - self.mean) / self.scale) @ self.coef + self.intercept[0]
        except FloatingPointError as error:
            raise ValueError("Laterality classifier normalization overflow") from error
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite laterality log odds")
        result = np.empty_like(z)
        positive = z >= 0
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp = np.exp(z[~positive])
        result[~positive] = exp / (1 + exp)
        return result

    def predict_array(self, pixels, *, source=None):
        try:
            frame = prepare_native(pixels)
        except ValueError as error:
            return {"source": str(source) if source else None, "laterality": None, "label": None,
                    "scores": None, "confidence": None, "needs_review": True,
                    "metadata": {"status": "invalid_image", "reason": str(error), "provenance": self.provenance}}
        p_right = float(self.score_features(self.extractor.extract(frame[None]))[0])
        confidence = max(p_right, 1 - p_right)
        return {"source": str(source) if source else None, "laterality": "right_hip" if p_right >= .5 else "left_hip",
                "label": int(p_right >= .5), "scores": {"left_hip": 1 - p_right, "right_hip": p_right},
                "confidence": confidence, "needs_review": confidence < .9,
                "metadata": {"status": "classified", "preprocessing": PREPROCESSING, "mirror_average": False,
                             "label_interpretation": "Agreement with provisional dataset side labels; verified anatomical side is unavailable",
                             "assumes_hip_input": True, "score_interpretation": "Uncalibrated score, not an anatomy or OOD detector",
                             "provenance": self.provenance}}
