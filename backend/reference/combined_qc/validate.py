"""Study-grouped validation of the deployed compact model recipes.

Usage: python -m combined_qc.validate --ci

This command fits temporary cross-validation classifiers, never overwrites the
production checkpoints, and uses verified production ONNX feature extractors.
The final classifiers fitted on every label cannot validate their own training
images; training the final deployment is a separate ``train_all`` operation.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Global DXA manifest/splits folder")
    parser.add_argument("--source-data", type=Path, help="Native DICOM fallback root")
    parser.add_argument("--spine-data", type=Path, help="Corrected spine label folder")
    parser.add_argument("--hip-checkpoints", type=Path)
    parser.add_argument("--spine-checkpoints", type=Path)
    parser.add_argument("--router-checkpoints", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-cache", action="store_true", help="Recompute frozen features")
    parser.add_argument("--skip-fixed-rules", action="store_true",
                        help="Skip the separately labelled development-only spine rule table")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()
    from .validation import validate
    validate(data=args.data_dir, source_data=args.source_data, spine_data=args.spine_data,
             hip_checkpoints=args.hip_checkpoints, spine_checkpoints=args.spine_checkpoints,
             router_checkpoints=args.router_checkpoints, output_dir=args.output_dir,
             resume=not args.no_cache, include_fixed_rules=not args.skip_fixed_rules,
             verbose=args.verbose, ci=args.ci)


if __name__ == "__main__":
    main()
