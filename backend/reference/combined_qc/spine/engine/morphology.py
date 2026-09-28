"""Fixed image descriptors for empirical DXA quality-control experiments.

These descriptors summarize native grayscale pixels and their dimensions. They
do not detect clinical anatomy, produce new labels, recover operator ROIs, or
convert pixel geometry to millimetres. No parameter is fitted across images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _binned_profile(profile: np.ndarray, bins: int) -> np.ndarray:
    """Area-average a piecewise-constant profile into fixed relative-position bins.

    Fractional boundaries handle tiny images without empty bins. This operation
    keeps the frame extent and does not center, crop, or align the foreground.
    """
    integral = np.concatenate(([0.0], np.cumsum(profile, dtype=np.float64)))
    edges = np.linspace(0.0, len(profile), bins + 1)
    at_edges = np.interp(edges, np.arange(len(profile) + 1), integral)
    return np.diff(at_edges) / (len(profile) / bins)


def _largest_zero_rectangle_fraction(zero: np.ndarray) -> float:
    """Largest axis-aligned all-zero rectangle, normalized by total frame area.

    This includes ordinary background. It is only a pixel descriptor, never an
    assertion that the region was redacted or is an imaging artifact.
    """
    height, width = zero.shape
    heights = np.zeros(width + 1, dtype=np.int64)  # Final zero is a sentinel.
    largest = 0
    for row in zero:
        heights[:-1] = np.where(row, heights[:-1] + 1, 0)
        stack: list[int] = []
        for col in range(width + 1):
            while stack and heights[stack[-1]] > heights[col]:
                bar = stack.pop()
                start = stack[-1] + 1 if stack else 0
                largest = max(largest, int(heights[bar]) * (col - start))
            stack.append(col)
    return largest / (height * width)


def _weighted_moments(a: np.ndarray, threshold: float) -> dict[str, float]:
    """Moments of max(intensity-threshold, 0), in native pixel orientation.

    Centroids use fractional frame coordinates. Covariances use isotropic pixel
    coordinates divided by max(H,W), so non-square frames are not stretched when
    computing axis orientation. Double-angle components avoid an arbitrary sign
    choice for an undirected principal axis. An isotropic/empty image has no axis
    and receives zero for both orientation components.
    """
    h, w = a.shape
    weights = np.maximum(a - threshold, 0.0)
    mass = float(weights.sum())
    names = ("centroid_x", "centroid_y", "variance_x", "variance_y", "covariance_xy",
             "axis_cos2", "axis_sin2", "eccentricity", "skew_x", "skew_y")
    if mass <= 0:
        return dict.fromkeys(names, 0.0)
    weights = weights / mass
    column_mass, row_mass = weights.sum(axis=0), weights.sum(axis=1)
    x_pixels, y_pixels = np.arange(w) + 0.5, np.arange(h) + 0.5
    cx, cy = float(column_mass @ x_pixels), float(row_mass @ y_pixels)
    dx, dy = (x_pixels - cx) / max(h, w), (y_pixels - cy) / max(h, w)
    vx, vy = float(column_mass @ (dx ** 2)), float(row_mass @ (dy ** 2))
    cov = float(dy @ weights @ dx)
    anisotropy = float(np.hypot(vx - vy, 2 * cov))
    if anisotropy > 1e-12:
        cos2, sin2 = (vx - vy) / anisotropy, 2 * cov / anisotropy
    else:
        cos2, sin2 = 0.0, 0.0
    major = (vx + vy + anisotropy) / 2
    minor = max(0.0, (vx + vy - anisotropy) / 2)
    eccentricity = np.sqrt(max(0.0, 1.0 - minor / major)) if major > 1e-12 else 0.0
    # Fixed regularization/clipping keeps very sparse, near-point shapes finite.
    skew_x = np.clip(float(column_mass @ (dx ** 3)) / max(vx, 1e-8) ** 1.5, -50, 50)
    skew_y = np.clip(float(row_mass @ (dy ** 3)) / max(vy, 1e-8) ** 1.5, -50, 50)
    return dict(zip(names, (cx / w, cy / h, vx, vy, cov, cos2, sin2,
                            float(eccentricity), float(skew_x), float(skew_y))))


def _image_features(array: np.ndarray) -> dict[str, float]:
    if array.ndim != 2 or array.dtype != np.uint8 or min(array.shape) < 1:
        raise ValueError("Morphology requires a nonempty two-dimensional uint8 image")
    h, w = array.shape
    a = array.astype(np.float64) / 255.0
    features: dict[str, float] = {
        "native_height": float(h), "native_width": float(w), "native_aspect_h_over_w": h / w,
        "intensity_mean": float(a.mean()), "intensity_std": float(a.std()),
        "intensity_min": float(a.min()), "intensity_max": float(a.max()),
    }
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        features[f"intensity_q{int(q * 100):02d}"] = float(np.quantile(a, q))
    histogram = np.bincount(np.minimum((a * 8).astype(np.int64), 7).ravel(), minlength=8)
    for i, value in enumerate(histogram / a.size):
        features[f"intensity_hist8_{i:02d}"] = float(value)

    border_h, border_w = max(1, int(np.ceil(h * 0.05))), max(1, int(np.ceil(w * 0.05)))
    for threshold in (0.10, 0.40, 0.70):
        mask = a > threshold
        prefix = f"above_{int(threshold * 100):02d}"
        features[prefix + "_fraction"] = float(mask.mean())
        for side, values in (("top", mask[:border_h]), ("bottom", mask[-border_h:]),
                             ("left", mask[:, :border_w]), ("right", mask[:, -border_w:])):
            features[f"{prefix}_border5pct_{side}"] = float(values.mean())

    for threshold in (0.10, 0.50):
        prefix = f"weighted_above_{int(threshold * 100):02d}"
        for name, value in _weighted_moments(a, threshold).items():
            features[f"{prefix}_{name}"] = value

    foreground = a > 0.10
    ys, xs = np.nonzero(foreground)
    margins = (float(ys.min()) / h, float(h - 1 - ys.max()) / h,
               float(xs.min()) / w, float(w - 1 - xs.max()) / w) if len(ys) else (1., 1., 1., 1.)
    for name, value in zip(("top", "bottom", "left", "right"), margins):
        features[f"above_10_margin_{name}"] = value

    for axis_name, axis in (("x", 0), ("y", 1)):
        for name, source, bins in (("intensity", a, 16), ("above_10", foreground, 8)):
            for i, value in enumerate(_binned_profile(source.mean(axis=axis), bins)):
                features[f"profile_{name}_{axis_name}_{bins}_{i:02d}"] = float(value)

    # Derivatives in native pixels, with edge replication rather than a new crop.
    p = np.pad(a, 1, mode="edge")
    gx, gy = (p[1:-1, 2:] - p[1:-1, :-2]) / 2, (p[2:, 1:-1] - p[:-2, 1:-1]) / 2
    gradient = np.hypot(gx, gy)
    laplacian = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4 * a
    local_mean = sum(p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9.0
    residual = a - local_mean
    features.update({
        "gradient_mean": float(gradient.mean()), "gradient_std": float(gradient.std()),
        "gradient_q90": float(np.quantile(gradient, 0.9)),
        "gradient_abs_x_mean": float(np.abs(gx).mean()), "gradient_abs_y_mean": float(np.abs(gy).mean()),
        "laplacian_abs_mean": float(np.abs(laplacian).mean()), "laplacian_std": float(laplacian.std()),
        "local_residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
        "gradient_above_015_fraction": float((gradient > 0.15).mean()),
    })

    zero = array == 0
    top, bottom = h // 4, max(h // 4 + 1, h - h // 4)
    left, right = w // 4, max(w // 4 + 1, w - w // 4)
    black_blocks = (zero[:-1, :-1] & zero[1:, :-1] & zero[:-1, 1:] & zero[1:, 1:]) if min(h, w) > 1 else zero
    features.update({
        "exact_zero_fraction": float(zero.mean()),
        "exact_zero_center_half_fraction": float(zero[top:bottom, left:right].mean()),
        "exact_zero_largest_rectangle_fraction": _largest_zero_rectangle_fraction(zero),
        "exact_zero_2x2_blocks_fraction": float(black_blocks.mean()),
    })
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Nonfinite morphology descriptor")
    return features


def extract_morphology(records: list[dict], dataset_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Return N×120 float32 features and their ordered names, preserving row order.

    Every supplied record is processed, including demo if present; this function
    does not inspect eligibility, labels, anatomy/side, study IDs or other
    metadata. Only ``native_path`` is read from a record. Width and height come
    from the decoded PNG, not the manifest. No train/demo statistics are fitted.

    ``exact_zero_*`` describes black pixels that may be ordinary background;
    these names do not assert detection of redaction or a quality violation.
    Empty input returns a (0,120) matrix with the same feature schema.
    """
    root = Path(dataset_dir).resolve()
    names = list(_image_features(np.zeros((1, 1), dtype=np.uint8)))
    rows = []
    for record in records:
        path = (root / record["native_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native image path escapes dataset directory")
        with Image.open(path) as image:
            if image.mode != "L" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Native image must be single-frame 8-bit grayscale")
            array = np.asarray(image).copy()
        descriptor = _image_features(array)
        if list(descriptor) != names:
            raise ValueError("Morphology feature schema changed between images")
        rows.append(list(descriptor.values()))
    values = np.asarray(rows, dtype=np.float32).reshape(len(records), len(names))
    if not np.isfinite(values).all():
        raise ValueError("Morphology values cannot be represented as finite float32")
    return values, names
