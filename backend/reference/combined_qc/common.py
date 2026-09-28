"""Shared, quiet-by-default interfaces for the two independent QC modules."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import uuid
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import os
import tempfile
import warnings

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = frozenset({".png", ".dcm", ".dicom"})


@contextmanager
def quiet_output(enabled=False):
    """Silence library diagnostics unless progress was explicitly enabled.

    Native descriptors are restored even on errors; exceptions still propagate.
    Like redirect_stdout, this context is intended for sequential CLI/API calls.
    """
    if enabled:
        yield
        return
    with warnings.catch_warnings(), tempfile.TemporaryFile() as native:
        warnings.simplefilter("ignore")
        sys.stdout.flush()
        sys.stderr.flush()
        saved = []
        try:
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(native.fileno(), descriptor)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
        finally:
            for descriptor, original in saved:
                os.dup2(original, descriptor)
                os.close(original)


class Progress:
    """Plain, noninteractive progress usable locally and in CI logs."""
    def __init__(self, region: str, enabled: bool = False):
        self.region, self.enabled = region, bool(enabled)

    def update(self, stage: str, message: str, current=None, total=None) -> None:
        if self.enabled:
            count = "" if current is None else f" {current}/{total}" if total is not None else f" {current}"
            print(f"[{self.region}] {stage}{count}: {message}", file=sys.stderr, flush=True)


def resolve_checkpoint_paths(region: str, checkpoints_dir=None) -> tuple[Path, Path, Path]:
    if region not in {"hip", "spine"}:
        raise ValueError("Unknown QC region")
    base = Path(checkpoints_dir).resolve() if checkpoints_dir is not None else ROOT / region / "checkpoints"
    return base, base / "source", base / "runtime"


def gather_sources(source, recursive: bool = False) -> tuple[list[Path], bool]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Expected a native grayscale PNG or DICOM: {path}")
        return [path], False
    if not path.is_dir():
        raise FileNotFoundError(path)
    iterator = path.rglob("*") if recursive else path.iterdir()
    files = [p.resolve() for p in sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)]
    if not files:
        raise ValueError(f"No PNG or DICOM images in {path}")
    return files, True


def image_payload(image) -> dict:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {"encoding": "base64_png", "mode": image.mode, "width": image.width,
            "height": image.height, "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


def write_json(path: Path, value) -> None:
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=convert) + "\n", encoding="utf-8")


def publish_directory(staging: Path, destination: Path) -> None:
    """Publish a validated complete runtime, restoring the old one on failure."""
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"Runtime destination must be a real directory: {destination}")
    backup = destination.with_name(destination.name + ".backup-" + uuid.uuid4().hex)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existed = destination.exists()
    if existed:
        destination.rename(backup)
    try:
        staging.rename(destination)
    except BaseException:
        if existed:
            backup.rename(destination)
        raise
    if existed:
        shutil.rmtree(backup)


def run_cli(region: str) -> None:
    """Identical command-line contract, with no implicit printing in CI."""
    import argparse
    from importlib import import_module

    parser = argparse.ArgumentParser(description=f"{region}: train, load, infer, convert")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "load-checkpoints", "infer", "convert"):
        command = sub.add_parser(action)
        command.add_argument("--checkpoints-dir", type=Path)
        command.add_argument("--device", default="auto")
        command.add_argument("--verbose", action="store_true", help="Print progress to stderr")
        command.add_argument("--ci", action="store_true", help="Explicitly enable CI progress logs")
        command.add_argument("--output", type=Path, help="Write JSON result (never dump images to terminal)")
        if action in {"train", "convert"}:
            command.add_argument("--data", type=Path)
        if action == "train":
            command.add_argument("--epochs", type=int)
            command.add_argument("--no-resume", action="store_true")
            command.add_argument("--no-convert", action="store_true")
        elif action == "infer":
            command.add_argument("source", type=Path)
            command.add_argument("--recursive", action="store_true")
            command.add_argument("--visualizations-dir", type=Path)
        elif action == "convert":
            command.add_argument("--verify-samples", type=int, default=2)
    args = parser.parse_args()
    api = import_module(f"combined_qc.{region}.pipeline")
    options = {key: getattr(args, key) for key in ("checkpoints_dir", "device", "verbose", "ci")}
    if args.action == "train":
        result = api.train(args.data, epochs=args.epochs, resume=not args.no_resume,
                           convert=not args.no_convert, **options)
    elif args.action == "load-checkpoints":
        model = api.load_checkpoints(**options)
        result = {"region": region, "checkpoints_dir": str(model.checkpoints_dir),
                  "status": "loaded", "metadata": getattr(model, "provenance", {})}
    elif args.action == "convert":
        result = import_module(f"combined_qc.{region}.convert_all").convert_all(
            data=args.data, verify_samples=args.verify_samples, **options)
    else:
        result = api.infer(args.source, recursive=args.recursive, **options)
        if args.visualizations_dir:
            args.visualizations_dir.mkdir(parents=True, exist_ok=True)
            rows = result if isinstance(result, list) else [result]
            for index, row in enumerate(rows):
                name = f"{index:04d}_{Path(row['source']).stem}.png"
                (args.visualizations_dir / name).write_bytes(
                    base64.b64decode(row["annotated_image"]["data"], validate=True))
    if args.output:
        write_json(args.output.resolve(), result)
