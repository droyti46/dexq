"""Infer one image and save the ready annotated PNG returned by ``infer``.

    python -m combined_qc.spine.examples.visualize_image --image /path/to/spine.png
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from ..pipeline import infer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/visualization_example")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = infer(args.image)
    json_path = output_dir / "inference.json"
    json_path.write_text(json.dumps(response, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")

    # The image is already rendered by infer. Saving it needs no source image,
    # plotting library, additional geometry, or model invocation.
    saved_response = json.loads(json_path.read_text(encoding="utf-8"))
    png_path = output_dir / "spine_visualization.png"
    png_path.write_bytes(base64.b64decode(saved_response["annotated_image"]["data"], validate=True))
    print(f"Inference response: {json_path}")
    print(f"Visualization: {png_path}")


if __name__ == "__main__":
    main()
