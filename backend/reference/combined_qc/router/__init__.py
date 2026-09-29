"""Anatomy classification and a reusable, unified quality-control pipeline."""

from .conversion import convert_all
from .pipeline import train, load_checkpoints, infer

__all__ = ["train", "convert_all", "load_checkpoints", "infer", "QCPipeline", "RegionClassifier"]


def __getattr__(name):
    if name == "QCPipeline":
        from .coordinator import QCPipeline
        return QCPipeline
    if name == "RegionClassifier":
        from .runtime import RegionClassifier
        return RegionClassifier
    raise AttributeError(name)
