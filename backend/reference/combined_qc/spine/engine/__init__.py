"""Portable retained lumbar-spine quality pipeline, with offline model assets."""

from .geometry import measure_tilt
from .composition import combine_quality, compose_spine_result
from .utils import read_manifest

__all__ = ['SpinePipeline', 'read_manifest', 'measure_tilt', 'combine_quality', 'compose_spine_result']


def __getattr__(name: str):
    # Keep the historical public name without loading the retired six-model
    # implementation or importing PyTorch/joblib.
    if name == 'SpinePipeline':
        from ..portable_inference import PortableSpinePipeline
        return PortableSpinePipeline
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
