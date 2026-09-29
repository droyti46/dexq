"""Pinned ResNet34 landmark detector and unchanged decoding into native pixels."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .geometry import measure_tilt
from .utils import sha256 as sha

SOURCE_COMMIT = "b9fc05c215ea2b006564a3feb509634183a63f82"
WEIGHTS_SHA256 = "6a779e01b9a41601334e0a9541278fc557a95bd650c6c8de311204821509d19b"
CONFIDENCE = 0.2
STRIDE = 4

def load_model(source: Path, weights: Path):
    manifest_path = source / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != SOURCE_COMMIT:
        raise ValueError("Source commit is not the audited pin")
    hashes = {}
    for item in manifest["files"]:
        name = item["path"]
        if name.startswith("models/"):
            p = (source / name).resolve()
            if not p.is_relative_to(source.resolve()) or sha(p) != item["sha256"]:
                raise ValueError(f"Audited source hash mismatch: {name}")
            hashes[name] = sha(p)
    if sha(weights) != WEIGHTS_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the audited download")
    package_name = "_lct_vertebra_landmark_models_b9fc05c"
    spec = importlib.util.spec_from_file_location(package_name, source / "models/__init__.py",
                                                 submodule_search_locations=[str(source / "models")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    module_spec = importlib.util.spec_from_file_location(package_name + ".spinal_net", source / "models/spinal_net.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    model = module.SpineNet(heads={"hm": 1, "reg": 2, "wh": 8}, pretrained=False,
                            down_ratio=STRIDE, final_kernel=1, head_conv=256)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    incompatibilities = model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not incompatibilities.missing_keys and not incompatibilities.unexpected_keys
    model.eval()
    return model, {"source_commit": SOURCE_COMMIT, "source_files_sha256": hashes,
                   "source_manifest_sha256": sha(manifest_path), "weights_sha256": sha(weights),
                   "checkpoint_epoch": checkpoint.get("epoch"), "strict_load": True,
                   "weights_only_load": True, "parameters": sum(p.numel() for p in model.parameters()),
                   "state_dict_tensors": len(checkpoint["state_dict"])}

def preprocess(array: np.ndarray):
    """Resize the entire native frame directly to 512×512, without padding."""
    h, w = array.shape
    resized = np.asarray(Image.fromarray(array).resize((512, 512), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(np.repeat(resized[None], 3, axis=0).astype(np.float32) / 255.0 - 0.5)[None]
    transform = {"method": "direct_square_resize", "native_height": h, "native_width": w,
                 "input_height": 512, "input_width": 512, "resized_height": 512,
                 "resized_width": 512, "left": 0, "top": 0,
                 "scale_x": 512 / w, "scale_y": 512 / h,
                 "coordinate_mapping": "native_x=(input_x-left)/scale_x; native_y=(input_y-top)/scale_y"}
    return tensor, transform

def inverse_xy(xy, transform):
    xy = np.asarray(xy, dtype=float)
    return ((xy - np.array([transform["left"], transform["top"]])) /
            np.array([transform["scale_x"], transform["scale_y"]]))

def decode(raw, transform, threshold=5.0):
    """Use every >=0.2 local heatmap peak; never force a fixed vertebra count."""
    hm = np.asarray(raw["hm"])
    reg, wh = np.asarray(raw["reg"]), np.asarray(raw["wh"])
    if hm.ndim != 3 or hm.shape[0] != 1 or reg.shape != (2, *hm.shape[1:]) or wh.shape != (8, *hm.shape[1:]):
        raise ValueError("Unexpected model output shape")
    if not all(np.isfinite(v).all() for v in (hm, reg, wh)):
        raise ValueError("Nonfinite model output")
    pooled = F.max_pool2d(torch.from_numpy(hm)[None], kernel_size=3, stride=1, padding=1)[0].numpy()
    yy, xx = np.where((hm[0] == pooled[0]) & (hm[0] >= CONFIDENCE))
    candidates, discarded = [], []
    h, w = transform["native_height"], transform["native_width"]
    for y, x in zip(yy.tolist(), xx.tolist()):
        center_feature = np.array([x, y], dtype=float) + reg[:, y, x]
        center_input = center_feature * STRIDE
        native = inverse_xy(center_input, transform)
        corners_input = (center_feature[None] - wh[:, y, x].reshape(4, 2)) * STRIDE
        candidate = {"xy": native.tolist(), "confidence": float(hm[0, y, x]),
                     "input_xy": center_input.tolist(), "heatmap_peak_yx": [y, x],
                     "corners_xy": inverse_xy(corners_input, transform).tolist(),
                     "corner_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
                     "anatomical_level": None}
        if not (0 <= native[0] < w and 0 <= native[1] < h):
            candidate["discard_reason"] = "center_outside_native_image_or_in_padding"
            discarded.append(candidate)
        else:
            candidates.append(candidate)
    candidates.sort(key=lambda c: (c["xy"][1], c["xy"][0]))
    for index, candidate in enumerate(candidates):
        candidate["index_from_top"] = index
    reasons = ["ap_radiograph_to_dxa_transfer_not_anatomically_validated"]
    if discarded:
        reasons.append("out_of_bounds_centers_discarded")
    if len(candidates) > 12:
        reasons.append("more_than_12_candidate_centers")
    measurement = None
    if len(candidates) < 2:
        reasons.append("fewer_than_two_candidate_centers")
    elif candidates[-1]["xy"][1] <= candidates[0]["xy"][1]:
        reasons.append("endpoint_vertical_span_not_positive")
    else:
        measurement = measure_tilt(candidates[0]["xy"], candidates[-1]["xy"], threshold_deg=threshold)
        if measurement["dy_px"] < 0.5 * h:
            reasons.append("endpoint_span_below_half_image_height")
    return {"method": "pretrained_resnet34_heatmap_two_extreme_centers_v1",
            "config": {"confidence_threshold": CONFIDENCE, "nms_kernel": 3, "threshold_deg": threshold,
                       "forced_candidate_count": None, "maximum_candidate_review_threshold": 12,
                       "minimum_endpoint_span_review_fraction": 0.5},
            "image_width": w, "image_height": h,
            "coordinate_system": "native_pixels_x_right_y_down", "anatomical_landmarks_validated": False,
            "status": "needs_review" if measurement else "failed", "review_reasons": reasons,
            "measurement": measurement, "candidates": candidates, "discarded_candidates": discarded,
            "preprocessing": transform}
