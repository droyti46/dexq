"""Optional ONNX export and parity checks for the two frozen spine backbones.

This module exports only neural-network computation. The native-image PNG
reader, 320/512 pixel resizing, morphology features, the logistic regression
classifier, heatmap decoding, and geometric criteria remain in Python.

Requires ``onnx`` for export and ``onnxruntime`` for execution. The legacy
PyTorch exporter (``dynamo=False``) does not need ``onnxscript``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .engine.axis import measure_axis
from .engine.features import grid_pool, load_resnet18
from .engine.landmarks import decode, load_model, preprocess
from .engine.utils import read_grayscale, read_manifest, safe_path, sha256


ROOT = Path(__file__).resolve().parent
RESNET_NAME = "resnet18_features.onnx"
SPINENET_NAME = "spinenet_512.onnx"
OPSET = 17


class ResNet18Features(nn.Module):
    """uint8 `[N,320,320]` -> global `[N,512]`, spatial `[N,2048]`.

    The uint8 conversion, `/255`, channel triplication, and ImageNet
    normalization exactly match ``engine.features.extract_feature_banks``.
    Morphology features are intentionally not part of this neural graph.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
                             .reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
                             .reshape(1, 3, 1, 1))

    def forward(self, gray_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = gray_u8.to(torch.float32).unsqueeze(1) / 255.0
        rgb = gray.repeat(1, 3, 1, 1)
        feature_map = self.backbone((rgb - self.mean) / self.std)
        return F.adaptive_avg_pool2d(feature_map, 1).flatten(1), grid_pool(feature_map)


class SpineNetMaps(nn.Module):
    """float32 `[1,3,512,512]` -> hm/reg/wh maps in stable output order."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, spine_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.model(spine_input)
        return raw["hm"], raw["reg"], raw["wh"]


def _load_torch_models(root: Path, source_dir=None) -> tuple[ResNet18Features, SpineNetMaps, dict]:
    torch.set_num_threads(4)
    source_dir = Path(source_dir).resolve() if source_dir is not None else root / "checkpoints/source"
    backbone, resnet_info = load_resnet18(source_dir)
    spinenet, spine_info = load_model(
        root / "vendor/vertebra_landmark", source_dir / "vertebra_landmark.pth")
    return (ResNet18Features(backbone).cpu().eval(),
            SpineNetMaps(spinenet).cpu().eval(),
            {"resnet18": resnet_info, "spinenet": spine_info})


def export_models(output_dir: str | Path, *, root: str | Path = ROOT,
                  verify_samples: int = 0, source_dir=None, data=None) -> dict:
    """Export verified local weights to two self-contained ONNX files.

    ResNet18 has a symbolic batch dimension and fixed 320×320 spatial shape.
    SpineNet is static `[1,3,512,512]`, reflecting per-image heatmap decoding.
    No user data, preprocessing or sklearn classifier is embedded in ONNX.
    """
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install onnx to export the spine neural models") from exc
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resnet, spinenet, provenance = _load_torch_models(root, source_dir=source_dir)
    resnet_path, spinenet_path = output_dir / RESNET_NAME, output_dir / SPINENET_NAME

    with torch.inference_mode():
        torch.onnx.export(
            resnet, (torch.zeros((1, 320, 320), dtype=torch.uint8),), resnet_path,
            input_names=["gray_u8"], output_names=["global_512", "spatial_2048"],
            opset_version=OPSET, dynamo=False, external_data=False,
            dynamic_axes={"gray_u8": {0: "batch"}, "global_512": {0: "batch"},
                          "spatial_2048": {0: "batch"}},
        )
        torch.onnx.export(
            spinenet, (torch.zeros((1, 3, 512, 512), dtype=torch.float32),), spinenet_path,
            input_names=["spine_input"], output_names=["hm", "reg", "wh"],
            opset_version=OPSET, dynamo=False, external_data=False,
        )
    onnx.checker.check_model(str(resnet_path))
    onnx.checker.check_model(str(spinenet_path))
    report = {
        "opset": OPSET,
        "exporter": "torch.onnx.export(dynamo=False)",
        "resnet18": {"path": RESNET_NAME, "sha256": sha256(resnet_path),
                     "input": "gray_u8:uint8[batch,320,320]",
                     "outputs": ["global_512:float32[batch,512]",
                                 "spatial_2048:float32[batch,2048]"]},
        "spinenet": {"path": SPINENET_NAME, "sha256": sha256(spinenet_path),
                     "input": "spine_input:float32[1,3,512,512]",
                     "outputs": ["hm:float32[1,1,128,128]",
                                 "reg:float32[1,2,128,128]",
                                 "wh:float32[1,8,128,128]"]},
        "source_weights": provenance,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if verify_samples:
        report["parity"] = verify_parity(output_dir, root=root, sample_count=verify_samples, source_dir=source_dir, data=data)
    return report


class ONNXSpineModels:
    """CPU ONNX Runtime sessions with strict input-shape and dtype checks."""

    def __init__(self, model_dir: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime to run the spine ONNX models") from exc
        model_dir = Path(model_dir).resolve()
        manifest = json.loads((model_dir / "export_manifest.json").read_text())
        if manifest["opset"] != OPSET:
            raise ValueError("Unexpected ONNX opset in export manifest")
        for part in ("resnet18", "spinenet"):
            path = safe_path(model_dir, manifest[part]["path"], manifest[part]["sha256"])
            if part == "resnet18":
                self.resnet = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            else:
                self.spinenet = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def resnet_features(self, gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return neural feature arrays; append morphology outside this call."""
        image = np.asarray(gray_u8)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[1:] != (320, 320) or len(image) == 0:
            raise ValueError("Expected uint8 [batch,320,320] prepared grayscale images")
        global_512, spatial_2048 = self.resnet.run(
            ["global_512", "spatial_2048"], {"gray_u8": np.ascontiguousarray(image)})
        if global_512.shape != (len(image), 512) or spatial_2048.shape != (len(image), 2048):
            raise ValueError("Unexpected ResNet18 ONNX output shape")
        return global_512, spatial_2048

    def spine_maps(self, spine_input: np.ndarray) -> dict[str, np.ndarray]:
        """Return raw maps; call existing ``decode`` and ``measure_axis`` in Python."""
        image = np.asarray(spine_input)
        if image.dtype != np.float32 or image.shape != (1, 3, 512, 512):
            raise ValueError("Expected float32 [1,3,512,512] from landmarks.preprocess")
        hm, reg, wh = self.spinenet.run(
            ["hm", "reg", "wh"], {"spine_input": np.ascontiguousarray(image)})
        expected = {"hm": (1, 1, 128, 128), "reg": (1, 2, 128, 128),
                    "wh": (1, 8, 128, 128)}
        raw = {"hm": hm, "reg": reg, "wh": wh}
        if any(raw[key].shape != shape for key, shape in expected.items()):
            raise ValueError("Unexpected SpineNet ONNX output shape")
        return raw


def verify_parity(model_dir: str | Path, *, root: str | Path = ROOT,
                  sample_count: int = 3, atol: float = 1e-3, rtol: float = 1e-4, source_dir=None, data=None) -> dict:
    """Compare raw tensors and decoded axis endpoints on real bundled images.

    This is a small smoke test, not validation over all 100 images. For a
    production migration, run the same comparison on all 100 and compare the
    artifact scores and final three decisions too.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    root = Path(root).resolve()
    sessions = ONNXSpineModels(model_dir)
    resnet, spinenet, _ = _load_torch_models(root, source_dir=source_dir)
    data_dir = Path(data).resolve() if data is not None else root / "data"
    rows = read_manifest(data_dir / "manifest.jsonl")[:sample_count]
    if len(rows) != sample_count:
        raise ValueError("Fewer manifest images than requested parity samples")
    cases = []
    prepared_batch = []
    with torch.inference_mode():
        for row in rows:
            native = read_grayscale(safe_path(data_dir, row["native_path"], row["native_sha256"]))
            prepared = read_grayscale(safe_path(data_dir, row["image_path"], row["image_sha256"]))
            prepared_batch.append(prepared)
            gray = np.ascontiguousarray(prepared[None])
            ref_global, ref_spatial = [v.numpy() for v in resnet(torch.from_numpy(gray))]
            got_global, got_spatial = sessions.resnet_features(gray)
            tensor, transform = preprocess(native)
            source_maps = dict(zip(("hm", "reg", "wh"),
                                   [v.numpy() for v in spinenet(tensor)]))
            onnx_maps = sessions.spine_maps(tensor.numpy())
            checked = {"image_id": row["image_id"], "max_absolute_error": {}}
            for name, source, converted in (
                ("global_512", ref_global, got_global),
                ("spatial_2048", ref_spatial, got_spatial),
                *((name, source_maps[name], onnx_maps[name]) for name in ("hm", "reg", "wh")),
            ):
                checked["max_absolute_error"][name] = float(np.max(np.abs(source - converted)))
                if not np.allclose(source, converted, atol=atol, rtol=rtol):
                    raise AssertionError(f"{row['image_id']}: ONNX {name} differs from PyTorch")
            source_detected = decode({k: v[0] for k, v in source_maps.items()}, transform)
            onnx_detected = decode({k: v[0] for k, v in onnx_maps.items()}, transform)
            source_points, onnx_points = source_detected["candidates"], onnx_detected["candidates"]
            if len(source_points) != len(onnx_points):
                raise AssertionError(f"{row['image_id']}: ONNX changed candidate count")
            center_error = max((float(np.max(np.abs(np.asarray(a["xy"]) - np.asarray(b["xy"]))))
                                for a, b in zip(source_points, onnx_points)), default=0.0)
            if center_error > 0.05:
                raise AssertionError(f"{row['image_id']}: ONNX center differs by {center_error:.3f} px")
            source_axis, onnx_axis = measure_axis(source_detected), measure_axis(onnx_detected)
            if source_axis["violation"] != onnx_axis["violation"]:
                raise AssertionError(f"{row['image_id']}: ONNX changed axis decision")
            checked.update(n_candidates=len(source_points), center_max_error_px=center_error,
                           axis_violation=source_axis["violation"])
            cases.append(checked)
        if len(prepared_batch) >= 2:
            # A batch of two verifies that the ResNet ONNX batch axis is real,
            # not merely a shape annotation on a batch-one export.
            batch = np.stack(prepared_batch[:2])
            reference_batch = [v.numpy() for v in resnet(torch.from_numpy(batch))]
            converted_batch = sessions.resnet_features(batch)
            for name, source, converted in zip(("global_512", "spatial_2048"),
                                               reference_batch, converted_batch):
                if not np.allclose(source, converted, atol=atol, rtol=rtol):
                    raise AssertionError(f"Dynamic-batch ONNX {name} differs from PyTorch")
    return {"n_images": len(cases), "atol": atol, "rtol": rtol,
            "resnet_dynamic_batch_2_checked": len(prepared_batch) >= 2,
            "max_center_error_px": max(c["center_max_error_px"] for c in cases),
            "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "checkpoints/runtime/onnx")
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    if args.action == "export":
        result = export_models(args.model_dir, verify_samples=args.sample_count)
    else:
        result = verify_parity(args.model_dir, sample_count=args.sample_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
