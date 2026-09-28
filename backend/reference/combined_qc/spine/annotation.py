"""Render the public spine inference result without recomputing geometry.

Only the embedded input PNG and the returned native-pixel coordinates are used.
This module does not open source images, load models, or change any decisions.
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import math

from PIL import Image, ImageDraw


POINT_COLOR = "#ff4f91"
GAP_COLOR = "#33d6cf"
AXIS_COLOR = "#ffce6e"


def _xy(value, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain an (x, y) coordinate")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name} must contain finite numeric coordinates")
    point = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in point):
        raise ValueError(f"{name} must contain finite numeric coordinates")
    return point


def annotate_image(payload: dict) -> dict:
    """Return one RGB PNG with vertebral centers, gap lines and the angle axis.

    ``payload`` must contain the public inference ``image`` and ``geometry``
    fields. All coordinates are drawn exactly as returned, at the native image
    resolution. Candidate corner polygons and pelvic proxies are not drawn.
    Missing detections are valid and simply leave their overlay absent.

    The returned dictionary is JSON-ready; ``data`` is the base64 PNG itself.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected a spine inference dictionary")
    source, geometry = payload.get("image"), payload.get("geometry")
    if not isinstance(source, dict) or not isinstance(geometry, dict):
        raise ValueError("The inference result must include image and geometry")
    if source.get("encoding") != "base64_png" or source.get("mode") != "L":
        raise ValueError("Expected the embedded grayscale base64 PNG")
    width, height = source.get("width"), source.get("height")
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Embedded image dimensions must be positive integers")
    if (geometry.get("image_width") != width
            or geometry.get("image_height") != height
            or geometry.get("coordinate_system") != "native_pixels_x_right_y_down"):
        raise ValueError("Image dimensions and native geometry must match")
    if not isinstance(source.get("data"), str):
        raise ValueError("Embedded image data must be base64 text")
    try:
        png = base64.b64decode(source["data"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid embedded base64 image") from exc
    try:
        with Image.open(BytesIO(png)) as original:
            if original.format != "PNG" or original.mode != "L" or original.size != (width, height):
                raise ValueError("PNG pixels do not match the embedded image metadata")
            original.load()
            image = original.convert("RGB")
    except (OSError, SyntaxError) as exc:
        raise ValueError("Invalid embedded PNG image") from exc

    candidates = geometry.get("vertebral_candidates", [])
    gaps = geometry.get("gap_lines", [])
    if not isinstance(candidates, list) or not isinstance(gaps, list):
        raise ValueError("Vertebral candidates and gap lines must be lists")
    centers = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each vertebral candidate must be a dictionary")
        centers.append(_xy(candidate.get("center_xy"), "Vertebral center"))
    gap_endpoints = []
    for gap in gaps:
        if not isinstance(gap, dict):
            raise ValueError("Each gap line must be a dictionary")
        endpoints = gap.get("endpoints_xy")
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
            raise ValueError("Each gap line requires two endpoints")
        gap_endpoints.append(tuple(_xy(p, "Gap endpoint") for p in endpoints))
    axis = geometry.get("axis_line")
    axis_endpoints = None
    if axis is not None:
        if not isinstance(axis, dict):
            raise ValueError("The angle axis must be a dictionary or None")
        axis_endpoints = (_xy(axis.get("top_xy"), "Axis top"),
                          _xy(axis.get("bottom_xy"), "Axis bottom"))

    # Draw at native resolution: the source is unchanged outside the overlays.
    scale = min(width, height)
    radius = max(2, round(scale / 250))
    line_width = max(2, round(scale / 260))
    draw = ImageDraw.Draw(image)
    for endpoints in gap_endpoints:
        draw.line(endpoints, fill=GAP_COLOR, width=line_width)
    if axis_endpoints is not None:
        draw.line(axis_endpoints, fill=AXIS_COLOR, width=line_width)
    for x, y in centers:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=POINT_COLOR)
    if axis_endpoints is not None:
        ring_radius = radius + max(2, radius)
        for x, y in axis_endpoints:
            draw.ellipse((x - ring_radius, y - ring_radius,
                          x + ring_radius, y + ring_radius),
                         outline=AXIS_COLOR, width=max(1, line_width // 2))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {
        "encoding": "base64_png", "mode": "RGB", "width": width, "height": height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "overlays": {
            "vertebral_points": {"color": POINT_COLOR, "count": len(centers)},
            "gap_lines": {"color": GAP_COLOR, "count": len(gap_endpoints)},
            "angle_axis": {"color": AXIS_COLOR, "available": axis_endpoints is not None},
        },
    }
