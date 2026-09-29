"""Diagnostic visualization using only the public hip inference result."""
from __future__ import annotations
import base64
from io import BytesIO
import math
from PIL import Image, ImageDraw
from ..common import image_payload


def _bbox(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Expected a four-coordinate diagnostic bounding box")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("Diagnostic box coordinates must be finite")
    x0, y0, x1, y1 = value
    if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
        raise ValueError("Diagnostic box is outside the native image")
    return x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)


def annotate_image(payload):
    """Draw the complete native photo, pixel-field box and compact label legend.

    The box is a brightness diagnostic, not a detected femur or anatomical ROI.
    It is returned and drawn independently of the predicted quality label.
    """
    image_info, geometry = payload.get("image"), payload.get("geometry")
    if not isinstance(image_info, dict) or not isinstance(geometry, dict):
        raise ValueError("Hip visualization requires image and geometry")
    if image_info.get("encoding") != "base64_png" or image_info.get("mode") != "L":
        raise ValueError("Expected the embedded grayscale PNG")
    width, height = image_info.get("width"), image_info.get("height")
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Embedded dimensions must be positive integers")
    if (geometry.get("image_width") != width or geometry.get("image_height") != height
            or geometry.get("coordinate_system") != "native_pixels_x_right_y_down"):
        raise ValueError("Hip image and geometry dimensions must match")
    try:
        data = base64.b64decode(image_info["data"], validate=True)
        with Image.open(BytesIO(data)) as source:
            if source.format != "PNG" or source.mode != "L" or source.size != (width, height):
                raise ValueError("Embedded PNG does not match its metadata")
            source.load()
            image = source.convert("RGB")
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("Invalid embedded hip PNG") from exc
    draw = ImageDraw.Draw(image)
    foreground = geometry.get("foreground_bbox")
    count = 0
    if foreground is not None:
        draw.rectangle(_bbox(foreground, width, height), outline="#33d6cf",
                       width=max(2, round(min(width, height) / 180)))
        count = 1
    panel = Image.new("RGB", (max(width, 280), height + 52), "#11151b")
    panel.paste(image, (0, 0))
    caption = ImageDraw.Draw(panel)
    labels = payload.get("labels", {})
    def text_label(value):
        return "?" if value is None else str(int(value))
    caption.text((6, height + 4), f"P: {text_label(labels.get('hip_positioning_rotation'))}   ROI: {text_label(labels.get('hip_roi'))}", fill="white")
    caption.text((6, height + 20), "Cyan: bright-pixel field boundary", fill="#33d6cf")
    caption.text((6, height + 36), "1 = QC violation; no anatomical ROI", fill="#c0c4ca")
    result = image_payload(panel)
    result["overlays"] = {"foreground_boxes": count, "foreground_color": "#33d6cf",
                          "interpretation": "brightness-based diagnostic field; no anatomical segmentation",
                          "native_image_width": width, "native_image_height": height}
    return result
