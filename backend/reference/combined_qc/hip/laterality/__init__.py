"""Production hip-side component; no training frameworks imported on load."""
__all__ = ['LateralityClassifier', 'train', 'export_runtime']


def __getattr__(name):
    if name == 'LateralityClassifier':
        from .runtime import LateralityClassifier
        return LateralityClassifier
    if name == 'train':
        from .training import train
        return train
    if name == 'export_runtime':
        from .conversion import export_runtime
        return export_runtime
    raise AttributeError(name)
