"""Local paths, integrity checks and prepared grayscale input helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_manifest(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def safe_path(root: str | Path, relative: str, expected_sha256: str | None = None) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes its data/model directory: {relative}')
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f'File checksum mismatch: {relative}')
    return path


def read_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'L' or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Expected a single-frame 8-bit grayscale PNG')
        return np.asarray(image).copy()


def letterbox320(array: np.ndarray) -> np.ndarray:
    """Exact preparation: bilinear resize, round-half-up, black centered padding."""
    if array.ndim != 2 or array.dtype != np.uint8 or min(array.shape) < 1:
        raise ValueError('Expected nonempty uint8 grayscale pixels')
    height, width = array.shape
    scale = 320 / max(height, width)
    resized_height = min(320, max(1, int(height * scale + 0.5)))
    resized_width = min(320, max(1, int(width * scale + 0.5)))
    top, left = (320 - resized_height) // 2, (320 - resized_width) // 2
    resized = Image.fromarray(array).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new('L', (320, 320), color=0)
    canvas.paste(resized, (left, top))
    return np.asarray(canvas).copy()
