"""V13 hip graphs for conversion only; inference never imports this module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def validate_logistic_payload(payload, feature_count):
    """Reject malformed numerical calibration before it enters an ONNX graph."""
    if payload.get("classes") != [0, 1]:
        raise ValueError("Hip calibration classes must be [0, 1]")
    arrays = {
        "scaler_mean": ((feature_count,), payload["scaler_mean"]),
        "scaler_scale": ((feature_count,), payload["scaler_scale"]),
        "coef": ((1, feature_count), payload["coef"]),
        "intercept": ((1,), payload["intercept"]),
    }
    result = {}
    for name, (shape, value) in arrays.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid hip calibration {name}: expected {shape}")
        result[name] = array
    if np.any(result["scaler_scale"] <= 0):
        raise ValueError("Hip calibration scales must be positive")
    return result


def checked_threshold(value):
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("Hip threshold must be finite and strictly between zero and one")
    return value


class HipEncoder(nn.Module):
    """Exact checkpoint architecture initialized from a local config, offline."""

    def __init__(self, config_path):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(str(Path(config_path)), local_files_only=True)
        self.backbone = AutoModel.from_config(config)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(8, 16), nn.LayerNorm(16), nn.SiLU(), nn.Dropout(0.35),
        )
        fused_size = config.hidden_sizes[-1] + 16
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_size), nn.Dropout(0.35), nn.Linear(fused_size, 128),
            nn.SiLU(), nn.Dropout(0.35),
        )
        self.positioning_rotation_head = nn.Linear(128, 1)
        self.roi_head = nn.Linear(128, 1)

    def forward(self, pixels, geometry):
        image_features = torch.flatten(self.backbone(pixel_values=pixels).pooler_output, 1)
        fused = self.fusion(torch.cat((image_features, self.geometry_encoder(geometry)), 1))
        return torch.cat((self.positioning_rotation_head(fused), self.roi_head(fused)), 1)


def model_from_state(state_dict, config_path):
    model = HipEncoder(config_path)
    model.load_state_dict(state_dict, strict=True)
    return model.float().cpu().eval()


class PositioningGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels, geometry):
        return torch.sigmoid(self.model(pixels, geometry)[:, 0:1])


class NumericLogistic(nn.Module):
    def __init__(self, payload, feature_count):
        super().__init__()
        arrays = validate_logistic_payload(payload, feature_count)
        for name, value in arrays.items():
            self.register_buffer(name, torch.from_numpy(value))

    def forward(self, features):
        scaled = (features.double() - self.scaler_mean) / self.scaler_scale
        return torch.sigmoid(scaled @ self.coef.T + self.intercept)


def tensor_logit(probability):
    probability = torch.clamp(probability.double(), 1e-6, 1.0 - 1e-6)
    return torch.log(probability / (1.0 - probability))


class ROIGraph(nn.Module):
    """Four FP32 neural members and the original two FP64 logistic stages."""

    def __init__(self, models, thresholds, calibration):
        super().__init__()
        if len(models) != 4 or len(thresholds) != 4:
            raise ValueError("V13 ROI requires exactly four neural members per outer fold")
        self.models = nn.ModuleList(models)
        values = np.clip([checked_threshold(value) for value in thresholds], 1e-6, 1.0 - 1e-6)
        self.register_buffer("threshold_logits", torch.from_numpy(np.log(values / (1.0 - values))))
        self.metadata = NumericLogistic(calibration["metadata_model"], 7)
        self.stacker = NumericLogistic(calibration["stacker_model"], 2)

    def forward(self, pixels, geometry, acquisition):
        raw = torch.cat([
            torch.sigmoid(model(pixels, geometry)[:, 1:2]) for model in self.models
        ], dim=1)
        centered = torch.sigmoid(tensor_logit(raw) - self.threshold_logits)
        neural_probability = torch.mean(centered, dim=1, keepdim=True)
        metadata_probability = self.metadata(acquisition)
        features = torch.cat((tensor_logit(neural_probability), tensor_logit(metadata_probability)), 1)
        return self.stacker(features)


def numpy_logistic(payload, features):
    arrays = validate_logistic_payload(payload, np.asarray(features).shape[1])
    logits = ((features - arrays["scaler_mean"]) / arrays["scaler_scale"]) @ arrays["coef"].T
    logits = logits + arrays["intercept"]
    return np.exp(-np.logaddexp(0.0, -logits))


def numpy_roi_reference(graph, inputs):
    """Independent numerical reference used to check exported calibration math."""
    pixels, geometry, acquisition = inputs
    with torch.inference_mode():
        raw = np.column_stack([
            torch.sigmoid(model(pixels, geometry)[:, 1]).numpy().astype(np.float64)
            for model in graph.models
        ])
    raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
    centered_logit = np.log(raw / (1.0 - raw)) - graph.threshold_logits.numpy()
    neural = np.exp(-np.logaddexp(0.0, -centered_logit)).mean(axis=1, keepdims=True)

    def payload(layer):
        return {"classes": [0, 1], **{
            name: getattr(layer, name).numpy() for name in ("scaler_mean", "scaler_scale", "coef", "intercept")
        }}

    metadata = numpy_logistic(payload(graph.metadata), acquisition.numpy())
    both = np.clip(np.concatenate((neural, metadata), axis=1), 1e-6, 1.0 - 1e-6)
    return numpy_logistic(payload(graph.stacker), np.log(both / (1.0 - both)))
