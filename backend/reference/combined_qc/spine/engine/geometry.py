"""Exact two-center angle in native image pixels; no fitting or angle tuning."""
import math
import numpy as np

def measure_tilt(top_xy, bottom_xy, *, threshold_deg: float = 5.0) -> dict:
    """Angle to upward vertical of the segment bottom -> top, in degrees.

    x grows right and y grows down. Image pixels are the metric; unknown DICOM
    pixel aspect ratio is not inferred. Call on native, unstretched coordinates.
    The sign is positive when the TOP endpoint is to the right of the bottom.
    A horizontal segment or inverted top/bottom ordering is invalid.
    """
    top, bottom = np.asarray(top_xy, dtype=float), np.asarray(bottom_xy, dtype=float)
    if top.shape != (2,) or bottom.shape != (2,) or not np.isfinite([top, bottom]).all():
        raise ValueError("Expected two finite (x, y) points")
    if not math.isfinite(threshold_deg) or not 0 <= threshold_deg < 90:
        raise ValueError("Threshold must be finite and in [0, 90)")
    dx, dy = float(top[0] - bottom[0]), float(bottom[1] - top[1])
    if dy <= 0:
        raise ValueError("Top endpoint must be strictly above bottom endpoint")
    signed = math.degrees(math.atan2(dx, dy))
    angle = abs(signed)
    # Floating point roundoff must not make exactly 5 degrees a violation.
    violation = angle > threshold_deg and not math.isclose(angle, threshold_deg, abs_tol=1e-10, rel_tol=0)
    return {"angle_deg": angle, "signed_angle_deg": signed,
            "threshold_deg": float(threshold_deg), "violation": bool(violation),
            "dx_px": dx, "dy_px": dy,
            "top_xy": top.tolist(), "bottom_xy": bottom.tolist()}
