"""Numeric, non-pickle format for the single spine artifact classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit, logit


N_FEATURES = 2680
FORMAT = "spine_single_logreg_artifacts_v1"


@dataclass(frozen=True)
class SingleLogReg:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    train_prevalence: float
    morphology_names: tuple[str, ...]

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        if (values.ndim != 2 or values.shape[1] != N_FEATURES
                or not np.isfinite(values).all()):
            raise ValueError("Expected finite [N,2680] artifact features")
        dtype = np.float32 if values.dtype == np.float32 else np.float64
        scaled = np.array(values, dtype=dtype, copy=True)
        scaled -= self.mean.astype(dtype, copy=False)
        scaled /= self.scale.astype(dtype, copy=False)
        return np.asarray(scaled @ self.coef + self.intercept, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Apply the retained pipeline's fixed prior correction."""
        raw = self.decision_function(features)
        return np.asarray(expit(raw + 0.5 * logit(self.train_prevalence)), dtype=np.float64)


def save_numeric_model(path: str | Path, pipeline, *, train_prevalence: float,
                       morphology_names: list[str]) -> None:
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    metadata = {
        "format": FORMAT, "n_features": N_FEATURES,
        "train_prevalence": float(train_prevalence),
        "morphology_feature_names": morphology_names,
        "feature_parts": ["resnet_global_512", "native_morphology_120", "resnet_spatial_2048"],
        "prior_strength": 0.5,
        "threshold": 0.5,
    }
    if (not 0 < train_prevalence < 1 or len(morphology_names) != 120
            or np.asarray(classifier.coef_).shape != (1, N_FEATURES)):
        raise ValueError("Unexpected fitted logistic model dimensions")
    payload = {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "coef": np.asarray(classifier.coef_[0], dtype=np.float64),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64),
        "metadata": np.frombuffer(json.dumps(metadata, ensure_ascii=False,
                                              sort_keys=True).encode("utf-8"), dtype=np.uint8),
    }
    np.savez_compressed(path, **payload)


def load_numeric_model(path: str | Path) -> SingleLogReg:
    """Load only numeric NPY arrays and UTF-8 JSON; pickle stays disabled."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"mean", "scale", "coef", "intercept", "metadata"}:
            raise ValueError("Unexpected single-logreg model archive members")
        encoded = data["metadata"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1 or len(encoded) > 65536:
            raise ValueError("Invalid single-logreg metadata")
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
        if (metadata.get("format") != FORMAT or metadata.get("n_features") != N_FEATURES
                or metadata.get("feature_parts") != [
                    "resnet_global_512", "native_morphology_120", "resnet_spatial_2048"]
                or metadata.get("prior_strength") != 0.5
                or metadata.get("threshold") != 0.5):
            raise ValueError("Unexpected single-logreg feature format")
        names = metadata.get("morphology_feature_names")
        prevalence = metadata.get("train_prevalence")
        if (not isinstance(names, list) or len(names) != 120
                or len(set(names)) != 120
                or not isinstance(prevalence, (int, float)) or not 0 < prevalence < 1):
            raise ValueError("Invalid single-logreg feature names or prevalence")
        arrays = {name: np.asarray(data[name]).copy()
                  for name in ("mean", "scale", "coef", "intercept")}
    for name, array, shape in (
        ("mean", arrays["mean"], (N_FEATURES,)),
        ("scale", arrays["scale"], (N_FEATURES,)),
        ("coef", arrays["coef"], (N_FEATURES,)),
        ("intercept", arrays["intercept"], (1,)),
    ):
        if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid numeric {name} array")
    if np.any(arrays["scale"] <= 0):
        raise ValueError("Non-positive StandardScaler scale")
    return SingleLogReg(arrays["mean"], arrays["scale"], arrays["coef"],
                        float(arrays["intercept"][0]), float(prevalence), tuple(names))
