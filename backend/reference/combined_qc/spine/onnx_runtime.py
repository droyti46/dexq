"""Inference-only ONNX Runtime loader for the two frozen spine networks.

This module deliberately imports neither PyTorch nor the export code. The
conversion manifest fixes model names, hashes, opset, and interface shapes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OPSET = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_path(directory: Path, spec: dict) -> Path:
    path = (directory / spec["path"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"ONNX model is missing or escapes its directory: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"ONNX model checksum mismatch: {path}")
    return path


class ONNXSpineModels:
    """Verified CPU ONNX sessions with fixed preprocessing interfaces."""

    def __init__(self, model_dir: str | Path, *, progress=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime to run the spine ONNX models") from exc
        model_dir = Path(model_dir).resolve()
        manifest = json.loads((model_dir / "export_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("opset") != OPSET:
            raise ValueError("Unexpected spine ONNX opset")
        resnet_path = _model_path(model_dir, manifest["resnet18"])
        spinenet_path = _model_path(model_dir, manifest["spinenet"])
        self.resnet = ort.InferenceSession(str(resnet_path), providers=["CPUExecutionProvider"])
        if progress is not None:
            progress.update("load", "resnet18_features.onnx loaded", 2, 3)
        self.spinenet = ort.InferenceSession(str(spinenet_path), providers=["CPUExecutionProvider"])
        if progress is not None:
            progress.update("load", "spinenet_512.onnx loaded", 3, 3)
        if ([(v.name, v.type) for v in self.resnet.get_inputs()] != [("gray_u8", "tensor(uint8)")]
                or [(v.name, v.type) for v in self.spinenet.get_inputs()]
                != [("spine_input", "tensor(float)")]):
            raise ValueError("Unexpected spine ONNX input names or types")
        if ([v.name for v in self.resnet.get_outputs()] != ["global_512", "spatial_2048"]
                or [v.name for v in self.spinenet.get_outputs()] != ["hm", "reg", "wh"]):
            raise ValueError("Unexpected spine ONNX output names")

    def resnet_features(self, gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract 512 global and 2048 spatial features from uint8 320² frames."""
        image = np.asarray(gray_u8)
        if (image.dtype != np.uint8 or image.ndim != 3
                or image.shape[1:] != (320, 320) or len(image) == 0):
            raise ValueError("Expected uint8 [batch,320,320] prepared grayscale images")
        global_512, spatial_2048 = self.resnet.run(
            ["global_512", "spatial_2048"], {"gray_u8": np.ascontiguousarray(image)})
        if (global_512.shape != (len(image), 512)
                or spatial_2048.shape != (len(image), 2048)
                or global_512.dtype != np.float32 or spatial_2048.dtype != np.float32
                or not np.isfinite(global_512).all() or not np.isfinite(spatial_2048).all()):
            raise ValueError("Invalid ResNet18 ONNX outputs")
        return global_512, spatial_2048

    def spine_maps(self, spine_input: np.ndarray) -> dict[str, np.ndarray]:
        """Return heatmap, center offsets, and corner offsets for decoding."""
        image = np.asarray(spine_input)
        if image.dtype != np.float32 or image.shape != (1, 3, 512, 512):
            raise ValueError("Expected float32 [1,3,512,512] spine input")
        hm, reg, wh = self.spinenet.run(
            ["hm", "reg", "wh"], {"spine_input": np.ascontiguousarray(image)})
        raw = {"hm": hm, "reg": reg, "wh": wh}
        expected = {"hm": (1, 1, 128, 128), "reg": (1, 2, 128, 128),
                    "wh": (1, 8, 128, 128)}
        if any(value.shape != expected[key] or value.dtype != np.float32
               or not np.isfinite(value).all() for key, value in raw.items()):
            raise ValueError("Invalid SpineNet ONNX outputs")
        return raw
