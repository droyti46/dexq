"""Central classifier API, matching the hip and spine function signatures."""
from pathlib import Path

from ..common import Progress, quiet_output


def train(data=None, *, checkpoints_dir=None, device="auto", epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    with quiet_output(verbose or ci):
        from .training import train as fit
        return fit(data, checkpoints_dir=checkpoints_dir, device=device, epochs=epochs,
                   verbose=verbose, ci=ci, resume=resume, convert=convert)


def convert_all(checkpoints_dir=None, *, data=None, device="auto", verbose=False,
                ci=False, verify_samples=2):
    from .conversion import convert_all as convert
    return convert(checkpoints_dir, data=data, device=device, verbose=verbose,
                   ci=ci, verify_samples=verify_samples)


def load_checkpoints(checkpoints_dir=None, *, device="auto", verbose=False, ci=False):
    if device not in {"auto", "cpu"}:
        raise ValueError("The router ONNX runtime uses CPU; use device='auto' or 'cpu'")
    with quiet_output(verbose or ci):
        from .runtime import RegionClassifier
        return RegionClassifier(checkpoints_dir, verbose=verbose, ci=ci)


def infer(source, *, checkpoints_dir=None, device="auto", verbose=False, ci=False,
          model=None, recursive=False):
    """Classify anatomy only; use QCPipeline.infer to also run quality control."""
    if device not in {"auto", "cpu"}:
        raise ValueError("The router ONNX runtime uses CPU")
    from .preprocessing import gather_sources, load_native
    from .runtime import RegionClassifier
    paths, folder = gather_sources(source, recursive)
    if model is not None and not isinstance(model, RegionClassifier):
        raise TypeError("Expected a router checkpoint handle")
    if model is not None and checkpoints_dir is not None and Path(checkpoints_dir).resolve() != model.checkpoints_dir:
        raise ValueError("Model handle and checkpoints_dir disagree")
    model = model if model is not None else load_checkpoints(
        checkpoints_dir, device=device, verbose=verbose, ci=ci)
    progress = Progress("router", verbose or ci)
    output = []
    for i, path in enumerate(paths, 1):
        output.append(model.predict_array(load_native(path), source=path))
        progress.update("classify", path.name, i, len(paths))
    return output if folder else output[0]
