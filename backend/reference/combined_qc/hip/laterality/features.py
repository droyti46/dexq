"""Orientation-preserving frozen ResNet18 embeddings, without mirror averaging."""
from pathlib import Path
import hashlib
import json

import numpy as np

from combined_qc.router.features import (
    DATA, RESNET_SHA, RESNET_STATE_SHA, FeatureExtractor as AnatomyExtractor,
    actual_source, load_record, prepare_native, records, sha256,
)

ROOT = Path(__file__).resolve().parent
HIP_ROOT = ROOT.parent
PROJECT = ROOT.parents[2]
DEFAULT_SOURCE = HIP_ROOT / 'checkpoints/source/laterality'
DEFAULT_RUNTIME = HIP_ROOT / 'checkpoints/runtime/laterality'
PREPROCESSING = "trim_exact_black_border_direct320_no_mirror_average_v1"
REPRESENTATIONS = {"global_512": 512, "spatial_2048": 2048, "global_spatial_2560": 2560}


def hip_records(data=DATA):
    rows = [row for row in records(data) if row["anatomical_region"] in {"left_hip", "right_hip"}]
    if len(rows) != 155:
        raise ValueError("Expected all 155 hip images; do not filter by image quality")
    return rows


def subset(features, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown feature representation")
    if representation == "global_512":
        selected = features[:, :512]
    elif representation == "spatial_2048":
        selected = features[:, 512:]
    else:
        selected = features
    return selected.astype(np.float64)


class FeatureExtractor:
    def __init__(self, path, expected_sha=None):
        self.session = AnatomyExtractor(path, expected_sha=expected_sha).session

    def extract(self, frames, batch_size=16):
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (320, 320) or not len(frames):
            raise ValueError("Expected uint8 [N,320,320] frames")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        parts = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            global_features, spatial_features = self.session.run(
                None, {"gray_u8": np.ascontiguousarray(batch)})
            if global_features.shape != (len(batch), 512) or spatial_features.shape != (len(batch), 2048):
                raise ValueError("Unexpected frozen ResNet18 feature dimensions")
            joined = np.concatenate([global_features, spatial_features], axis=1)
            if not np.isfinite(joined).all():
                raise ValueError("Nonfinite neural embeddings")
            parts.append(joined)
        return np.concatenate(parts).astype(np.float32)


def recipe_fingerprint():
    from importlib.metadata import version
    from combined_qc.router import features as shared_features
    from .bootstrap import export_recipe_sha256
    recipe = {
        "files": {name: sha256(ROOT / name) for name in ("features.py", "training.py", "bootstrap.py", "conversion.py")},
        "export_recipe_sha256": export_recipe_sha256(),
        "shared_preprocessing_code": sha256(Path(shared_features.__file__)),
        "libraries": {name: version(name) for name in ("numpy", "Pillow", "scikit-learn", "onnxruntime")},
        "preprocessing": PREPROCESSING,
    }
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
