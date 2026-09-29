"""V13 preprocessing shared by hip training, conversion and portable inference."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image


GEOMETRY_KEYS = (
    "letterbox_size", "original_height", "original_width", "resized_height",
    "resized_width", "pad_top", "pad_bottom", "pad_left", "pad_right",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("The hip bundle manifest must list its files")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Unsafe or missing bundle file: {name}")
        if not isinstance(expected, dict):
            raise ValueError(f"Invalid checksum entry: {name}")
        if path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return root, manifest


def load_native(path):
    """Read the native, unwindowed single-channel 8-bit export."""
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom
        ds = pydicom.dcmread(path)
        expected = {"PhotometricInterpretation": "MONOCHROME2", "SamplesPerPixel": 1,
                    "BitsAllocated": 8, "BitsStored": 8, "HighBit": 7, "PixelRepresentation": 0}
        if any(getattr(ds, key, None) != value for key, value in expected.items()):
            raise ValueError("DICOM must be single-channel uint8 MONOCHROME2")
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Multiframe DICOM is unsupported")
        pixels = ds.pixel_array
        if pixels.shape != (int(ds.Rows), int(ds.Columns)):
            raise ValueError("DICOM Rows/Columns mismatch")
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError("Hip images must be native 8-bit grayscale (mode L)")
            pixels = np.asarray(image).copy()
    else:
        raise ValueError("Expected a native grayscale image or .dcm/.dicom")
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or min(pixels.shape) == 0:
        raise ValueError("Input must be nonempty 2D uint8 grayscale")
    return pixels


def validate_geometry(geometry, size=320):
    """Reject invalid acquisition metadata before it reaches the ONNX graphs.

    Dimensions describe the original image, while padding describes its exact
    centered, bilinear, round-half-up letterbox. NaN must never turn into a
    silently negative quality decision.
    """
    if not isinstance(geometry, dict):
        raise ValueError("Geometry must be a dictionary describing the original image")
    values = {}
    for key in GEOMETRY_KEYS:
        value = geometry.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"Geometry {key} must be a finite integer")
        if not math.isfinite(float(value)) or float(value) != int(value):
            raise ValueError(f"Geometry {key} must be a finite integer")
        value = int(value)
        if value < 0 or value > 2**31 - 1:
            raise ValueError(f"Geometry {key} is out of range")
        values[key] = value
    if values["letterbox_size"] != size:
        raise ValueError(f"Geometry letterbox_size must be {size}")
    h, w = values["original_height"], values["original_width"]
    if h == 0 or w == 0:
        raise ValueError("Original image dimensions must be positive")
    scale = size / max(h, w)
    rh = min(size, max(1, int(h * scale + .5)))
    rw = min(size, max(1, int(w * scale + .5)))
    top, left = (size - rh) // 2, (size - rw) // 2
    expected = {"resized_height": rh, "resized_width": rw,
                "pad_top": top, "pad_bottom": size - rh - top,
                "pad_left": left, "pad_right": size - rw - left}
    if any(values[key] != value for key, value in expected.items()):
        raise ValueError("Geometry does not match a centered round-half-up letterbox of the original image")
    return values


def letterbox(array, size=320):
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.dtype != np.uint8 or min(array.shape) == 0:
        raise ValueError("Expected a nonempty native uint8 grayscale array")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Letterbox size must be a positive integer")
    height, width = array.shape
    scale = size / max(height, width)
    resized_height = min(size, max(1, int(height * scale + .5)))
    resized_width = min(size, max(1, int(width * scale + .5)))
    top, left = (size - resized_height) // 2, (size - resized_width) // 2
    resized = Image.fromarray(array).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(resized, (left, top))
    geometry = {"letterbox_size": size, "original_height": height, "original_width": width,
                "resized_height": resized_height, "resized_width": resized_width,
                "pad_top": top, "pad_bottom": size - resized_height - top,
                "pad_left": left, "pad_right": size - resized_width - left}
    return np.asarray(canvas, dtype=np.float32) / 255., geometry


def hip_geometry_features(pixels):
    # Fixed pixel-region proxies; these are not anatomical detections.
    h, w = pixels.shape
    whole = pixels[round(.05*h):round(.95*h), round(.05*w):round(.95*w)]
    superior_joint = pixels[round(.05*h):round(.35*h), round(.15*w):round(.85*w)]
    joint_field = pixels[round(.18*h):round(.68*h), round(.08*w):round(.92*w)]
    inferior_shaft = pixels[round(.65*h):round(.98*h), round(.20*w):round(.80*w)]
    lateral_left = pixels[round(.30*h):round(.75*h), round(.02*w):round(.40*w)]
    lateral_right = pixels[round(.30*h):round(.75*h), round(.60*w):round(.98*w)]
    denominator = float(whole.mean()) + 1e-6
    foreground = pixels > max(.02, float(np.quantile(pixels, .75)) * .10)
    ys, xs = np.where(foreground)
    bbox_h = (int(ys.max()) - int(ys.min()) + 1) / h if len(ys) else 0.
    bbox_w = (int(xs.max()) - int(xs.min()) + 1) / w if len(xs) else 0.
    shaft_foreground = foreground[round(.65*h):round(.98*h), round(.20*w):round(.80*w)]
    left_ratio = float(lateral_left.mean()) / denominator
    right_ratio = float(lateral_right.mean()) / denominator
    features = np.asarray([superior_joint.mean()/denominator, joint_field.mean()/denominator,
                           inferior_shaft.mean()/denominator, shaft_foreground.mean(),
                           max(left_ratio, right_ratio), abs(left_ratio-right_ratio),
                           bbox_h, bbox_w], dtype=np.float32)
    return np.nan_to_num(features, nan=0., posinf=3., neginf=0.).clip(0., 3.)


def acquisition_features(row):
    row = validate_geometry(row)
    size = float(row["letterbox_size"])
    width = float(row["original_width"])
    return np.asarray([row["resized_height"]/size, row["resized_width"]/size,
                       row["pad_top"]/size, row["pad_bottom"]/size,
                       row["pad_left"]/size, row["pad_right"]/size,
                       row["original_height"]/width], dtype=np.float64)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return 1. / (1. + np.exp(-np.clip(value, -80., 80.)))


def logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1. - 1e-6)
    return np.log(value / (1. - value))


def centered_score(probability, threshold):
    return sigmoid(logit(probability) - logit(threshold))
