"""PyTorch-free preprocessing and heatmap decoding for spine ONNX inference.

The 512×512 direct resize, 3×3 peak rule, native-pixel mapping, and review
status reproduce ``engine.landmarks`` without loading the detector checkpoint.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

from .engine.geometry import measure_tilt


CONFIDENCE = 0.2
STRIDE = 4


def preprocess(array: np.ndarray) -> tuple[np.ndarray, dict]:
    """Resize the complete native frame directly to float32 [1,3,512,512]."""
    if array.ndim != 2 or array.dtype != np.uint8 or min(array.shape) < 1:
        raise ValueError("Expected a nonempty 8-bit grayscale image")
    h, w = array.shape
    resized = np.asarray(Image.fromarray(array).resize((512, 512), Image.Resampling.BILINEAR))
    tensor = (np.repeat(resized[None], 3, axis=0).astype(np.float32) / 255.0 - 0.5)[None]
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


def decode(raw: dict[str, np.ndarray], transform: dict, threshold: float = 5.0) -> dict:
    """Decode every 3×3 heatmap peak at confidence >=0.2 to native pixels."""
    hm = np.asarray(raw["hm"])
    reg, wh = np.asarray(raw["reg"]), np.asarray(raw["wh"])
    if (hm.ndim != 3 or hm.shape[0] != 1
            or reg.shape != (2, *hm.shape[1:])
            or wh.shape != (8, *hm.shape[1:])):
        raise ValueError("Unexpected model output shape")
    if not all(np.isfinite(v).all() for v in (hm, reg, wh)):
        raise ValueError("Nonfinite model output")
    # PyTorch max_pool2d(kernel_size=3, stride=1, padding=1) uses -inf beyond
    # the image. SciPy's constant boundary gives the same values and ties.
    pooled = maximum_filter(hm[0], size=3, mode="constant", cval=-np.inf)
    yy, xx = np.where((hm[0] == pooled) & (hm[0] >= CONFIDENCE))
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
