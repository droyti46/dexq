"""Own all reusable model handles and dispatch each image by its anatomy."""
from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
import hashlib
from pathlib import Path
from threading import RLock

from ..common import Progress, gather_sources, write_json

NAMES = ("classifier", "hip", "spine")
ROOT = Path(__file__).resolve().parent


def _module(name):
    region = "router" if name == "classifier" else name
    return import_module(f"combined_qc.{region}.pipeline")


def _converter(name):
    if name == "classifier":
        return import_module("combined_qc.router.conversion")
    return import_module(f"combined_qc.{name}.convert_all")


def _data_options(data):
    if data is None:
        return {}
    if not isinstance(data, Mapping) or set(data) - set(NAMES):
        raise ValueError("data must map classifier, hip and/or spine to their dataset directories")
    return dict(data)


class QCPipeline:
    """One persistent instance per server worker or command-line session.

    Models load on first use or explicitly through load_checkpoints(). All
    inference uses converted ONNX/NPZ models. Calls on one instance are locked
    so checkpoint replacement and simultaneous first requests cannot race.
    Training and conversion do not automatically start during inference.
    """

    def __init__(self, checkpoints_dir=None, *, hip_checkpoints_dir=None,
                 spine_checkpoints_dir=None, device="auto", verbose=False, ci=False):
        if device not in {"auto", "cpu"}:
            raise ValueError("Unified ONNX inference supports device='auto' or 'cpu'")
        self.device, self.verbose, self.ci = device, bool(verbose), bool(ci)
        self.checkpoint_dirs = {
            "classifier": Path(checkpoints_dir).expanduser().resolve() if checkpoints_dir else ROOT / "checkpoints",
            "hip": Path(hip_checkpoints_dir).expanduser().resolve() if hip_checkpoints_dir else ROOT.parent / "hip/checkpoints",
            "spine": Path(spine_checkpoints_dir).expanduser().resolve() if spine_checkpoints_dir else ROOT.parent / "spine/checkpoints",
        }
        self.models = {}
        self.states = {name: {"status": "not_loaded", "checkpoints_dir": str(path)}
                       for name, path in self.checkpoint_dirs.items()}
        self._lock = RLock()

    def _flags(self, verbose, ci):
        return (self.verbose if verbose is None else bool(verbose),
                self.ci if ci is None else bool(ci))

    def _load(self, name, verbose, ci):
        if name not in self.models:
            handle = _module(name).load_checkpoints(
                self.checkpoint_dirs[name], device=self.device, verbose=verbose, ci=ci)
            self.models[name] = handle
            self.states[name].update(status="loaded", provenance=getattr(handle, "provenance", {}))
        else:
            Progress("router", verbose or ci).update("load", f"{name}: using models already in memory")
        return self.models[name]

    def load_checkpoints(self, regions=NAMES, *, verbose=None, ci=None):
        """Load selected modules once; default eagerly prepares all three."""
        names = (regions,) if isinstance(regions, str) else tuple(regions)
        if not names or any(name not in NAMES for name in names):
            raise ValueError("regions must contain classifier, hip and/or spine")
        verbose, ci = self._flags(verbose, ci)
        with self._lock:
            progress = Progress("router", verbose or ci)
            for i, name in enumerate(dict.fromkeys(names), 1):
                self._load(name, verbose, ci)
                progress.update("load", f"{name}: ready", i, len(set(names)))
        return self

    def _published(self, name, operation, report):
        self.models.pop(name, None)
        self.states[name].update(status="not_loaded", provenance={}, last_operation=operation,
                                 last_result=report)

    def _runtime_stamp(self, name):
        runtime = self.checkpoint_dirs[name] / "runtime"
        manifests = sorted(runtime.rglob("*manifest.json")) if runtime.is_dir() else []
        if not manifests:
            return None
        digest = hashlib.sha256()
        for manifest in manifests:
            digest.update(str(manifest.relative_to(runtime)).encode())
            digest.update(b"\0")
            digest.update(manifest.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _run_update(self, name, operation, callback):
        """A report-writing error can occur after a runtime was published."""
        before = self._runtime_stamp(name)
        try:
            return callback()
        except BaseException as error:
            if self._runtime_stamp(name) != before:
                self._published(name, operation + "_published_before_error",
                                {"error_type": type(error).__name__, "error": str(error)})
            raise

    def train_all(self, data=None, *, epochs=None, resume=True, convert=True,
                  verbose=None, ci=None):
        """Train classifier, hip and spine using each module's existing recipe.

        Data is a mapping because hip/router and spine have different label
        manifests. All backbones are frozen; every module accepts epochs=None or 1.
        A mapping can set each module explicitly.
        Successful conversion invalidates that module's old in-memory handle;
        its next use loads the newly published checkpoint.
        """
        data = _data_options(data)
        if isinstance(epochs, Mapping):
            if set(epochs) - set(NAMES):
                raise ValueError("Unknown module in epochs")
            epoch_options = dict(epochs)
        else:
            epoch_options = {"hip": epochs}
        verbose, ci = self._flags(verbose, ci)
        reports = {}
        with self._lock:
            progress = Progress("router", verbose or ci)
            for i, name in enumerate(NAMES, 1):
                progress.update("train_all", f"{name}: starting", i - 1, len(NAMES))
                report = self._run_update(name, "train", lambda: _module(name).train(
                    data.get(name), checkpoints_dir=self.checkpoint_dirs[name], device=self.device,
                    epochs=epoch_options.get(name), resume=resume, convert=convert,
                    verbose=verbose, ci=ci))
                reports[name] = report
                if convert:
                    self._published(name, "train_and_convert", report)
                else:
                    self.states[name].update(last_operation="train_source", last_result=report)
                progress.update("train_all", f"{name}: completed", i, len(NAMES))
            write_json(self.checkpoint_dirs["classifier"] / "global_training_report.json", reports)
        return reports

    def convert_all(self, data=None, *, verify_samples=2, verbose=None, ci=None):
        """Convert every saved source model, then discard stale loaded handles."""
        data = _data_options(data)
        if type(verify_samples) is not int or verify_samples < 1:
            raise ValueError("verify_samples must be a positive integer")
        verbose, ci = self._flags(verbose, ci)
        reports = {}
        with self._lock:
            progress = Progress("router", verbose or ci)
            for i, name in enumerate(NAMES, 1):
                progress.update("convert_all", f"{name}: starting", i - 1, len(NAMES))
                report = self._run_update(name, "convert", lambda: _converter(name).convert_all(
                    self.checkpoint_dirs[name], data=data.get(name), device=self.device,
                    verify_samples=verify_samples, verbose=verbose, ci=ci))
                reports[name] = report
                self._published(name, "convert", report)
                progress.update("convert_all", f"{name}: completed", i, len(NAMES))
            write_json(self.checkpoint_dirs["classifier"] / "global_conversion_report.json", reports)
        return reports

    def infer(self, source, *, recursive=False, verbose=None, ci=None):
        """Classify, route and return the complete profile inference result.

        A file returns one dict; a folder returns sorted dicts. The child
        result (including all geometry, metadata and visualization) is kept,
        with only an additional routing field. Unusable/uncertain images are
        returned for review without applying the wrong quality-control model.
        The hip module determines left/right automatically with its own
        persistent classifier; this routing layer never supplies a side.
        """
        paths, folder = gather_sources(source, recursive=recursive)
        verbose, ci = self._flags(verbose, ci)
        output = []
        with self._lock:
            classifier = self._load("classifier", verbose, ci)
            progress = Progress("router", verbose or ci)
            for i, path in enumerate(paths, 1):
                classification = _module("classifier").infer(
                    path, checkpoints_dir=self.checkpoint_dirs["classifier"], device=self.device,
                    verbose=verbose, ci=ci, model=classifier)
                region = classification.get("region")
                if classification.get("needs_review") or region is None:
                    status = classification.get("metadata", {}).get("status")
                    result = {"source": str(path), "region": region,
                              "status": "invalid_image" if status == "invalid_image" else "needs_review",
                              "routing": {**classification, "target_module": None},
                              "labels": {}, "scores": {}, "metadata": classification.get("metadata", {}),
                              "geometry": None, "annotated_image": None, "any_violation": None}
                else:
                    if region not in {"hip", "lumbar_spine"}:
                        raise ValueError(f"Classifier returned an unknown anatomy: {region!r}")
                    name = "hip" if region == "hip" else "spine"
                    handle = self._load(name, verbose, ci)
                    result = dict(_module(name).infer(
                        path, checkpoints_dir=self.checkpoint_dirs[name], device=self.device,
                        verbose=verbose, ci=ci, model=handle))
                    if "routing" in result:
                        raise ValueError("Profile inference already defines the reserved routing field")
                    result["routing"] = {**classification, "target_module": name}
                output.append(result)
                progress.update("infer", f"{path.name}: {result.get('status', result['routing']['target_module'])}", i, len(paths))
        return output if folder else output[0]
