"""Standalone lumbar-spine inference using ONNX and one numeric classifier.

Public exports are lazy so inference need not import PyTorch, scikit-learn or
joblib from optional conversion and research code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .portable_inference import PortableSpinePipeline
    SpinePipeline = PortableSpinePipeline
    from .pipeline import train, load_checkpoints, infer, run_spine

__all__ = ["SpinePipeline", "PortableSpinePipeline", "train", "load_checkpoints", "infer", "run_spine"]


def __getattr__(name: str):
    if name in ("SpinePipeline", "PortableSpinePipeline"):
        from .portable_inference import PortableSpinePipeline
        return PortableSpinePipeline
    if name in ("train", "load_checkpoints", "infer", "run_spine"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
