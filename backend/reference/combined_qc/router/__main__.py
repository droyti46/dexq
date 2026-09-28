"""Classifier operations and unified train/convert/infer commands."""
import argparse
import base64
from pathlib import Path

from ..common import write_json
from . import pipeline
from .coordinator import QCPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    actions = ("train", "convert", "load-checkpoints", "classify", "train-all", "convert-all", "load-all", "infer")
    for action in actions:
        p = sub.add_parser(action)
        p.add_argument("--checkpoints-dir", type=Path, help="Router/classifier checkpoints")
        p.add_argument("--device", default="auto", choices=("auto", "cpu"))
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--ci", action="store_true")
        p.add_argument("--output", type=Path)
        if action in {"train", "convert"}:
            p.add_argument("--data", type=Path)
        if action in {"train", "train-all"}:
            p.add_argument("--epochs", type=int, help="For train-all: hip DL epochs; ML backbones remain frozen")
            p.add_argument("--no-resume", action="store_true")
            p.add_argument("--no-convert", action="store_true")
        if action in {"convert", "convert-all"}:
            p.add_argument("--verify-samples", type=int, default=2)
        if action in {"train-all", "convert-all", "load-all", "infer"}:
            p.add_argument("--hip-checkpoints-dir", type=Path)
            p.add_argument("--spine-checkpoints-dir", type=Path)
        if action in {"train-all", "convert-all"}:
            for name in ("classifier", "hip", "spine"):
                p.add_argument(f"--{name}-data", type=Path)
        if action in {"classify", "infer"}:
            p.add_argument("source", type=Path)
            p.add_argument("--recursive", action="store_true")
        if action == "infer":
            p.add_argument("--visualizations-dir", type=Path)
    args = parser.parse_args(argv)
    options = {name: getattr(args, name) for name in ("checkpoints_dir", "device", "verbose", "ci")}
    if args.action == "train":
        result = pipeline.train(args.data, epochs=args.epochs, resume=not args.no_resume,
                                convert=not args.no_convert, **options)
    elif args.action == "convert":
        result = pipeline.convert_all(data=args.data, verify_samples=args.verify_samples, **options)
    elif args.action == "load-checkpoints":
        handle = pipeline.load_checkpoints(**options)
        result = {"status": "loaded", "provenance": handle.provenance}
    elif args.action == "classify":
        result = pipeline.infer(args.source, recursive=args.recursive, **options)
    else:
        qc = QCPipeline(hip_checkpoints_dir=args.hip_checkpoints_dir,
                        spine_checkpoints_dir=args.spine_checkpoints_dir, **options)
        if args.action in {"train-all", "convert-all"}:
            data = {name: getattr(args, name + "_data") for name in ("classifier", "hip", "spine")
                    if getattr(args, name + "_data") is not None}
        if args.action == "train-all":
            result = qc.train_all(data, epochs=args.epochs, resume=not args.no_resume,
                                  convert=not args.no_convert)
        elif args.action == "convert-all":
            result = qc.convert_all(data, verify_samples=args.verify_samples)
        elif args.action == "load-all":
            qc.load_checkpoints()
            result = {name: {"status": state["status"], "checkpoints_dir": state["checkpoints_dir"]}
                      for name, state in qc.states.items()}
        else:
            result = qc.infer(args.source, recursive=args.recursive)
            if args.visualizations_dir:
                args.visualizations_dir.mkdir(parents=True, exist_ok=True)
                for i, row in enumerate(result if isinstance(result, list) else [result]):
                    if row.get("annotated_image"):
                        name = f"{i:04d}_{Path(row['source']).stem}.png"
                        (args.visualizations_dir / name).write_bytes(base64.b64decode(
                            row["annotated_image"]["data"], validate=True))
    if args.output:
        write_json(args.output.resolve(), result)


if __name__ == "__main__":
    main()
