"""Hip quality control with portable inference and optional training tools."""

__all__ = ["train", "load_checkpoints", "infer", "HipONNXPredictor"]


def __getattr__(name):
    if name in {"train", "load_checkpoints", "infer"}:
        from . import pipeline
        return getattr(pipeline, name)
    if name == "HipONNXPredictor":
        from .runtime import HipONNXPredictor
        return HipONNXPredictor
    raise AttributeError(name)
