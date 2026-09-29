"""Offline ResNet18 and morphology features for the artifact classifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .morphology import _image_features
from .utils import sha256


def state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_resnet18(weights_dir: str | Path):
    target = Path(weights_dir) / 'resnet18.pt'
    metadata = json.loads(target.with_suffix('.json').read_text())
    if sha256(target) != metadata['file_sha256']:
        raise ValueError('Packaged ResNet18 checkpoint checksum mismatch')
    model = nn.Sequential(*list(models.resnet18(weights=None).children())[:-2])
    model.load_state_dict(torch.load(target, map_location='cpu', weights_only=True), strict=True)
    if state_digest(model) != metadata['state_sha256']:
        raise ValueError('Restored ResNet18 state differs')
    return model.eval(), metadata


def grid_pool(feature_map):
    """The original channel-major 2×2 grid pooling order, including odd bins."""
    height, width = feature_map.shape[-2:]
    bins = []
    for y in range(2):
        y0, y1 = y * height // 2, ((y + 1) * height + 1) // 2
        for x in range(2):
            x0, x1 = x * width // 2, ((x + 1) * width + 1) // 2
            bins.append(feature_map[:, :, y0:y1, x0:x1].mean(dim=(-2, -1)))
    return torch.stack(bins, dim=-1).flatten(1)


@torch.inference_mode()
def extract_feature_banks(model, image_ids: list[str], native_arrays: list[np.ndarray],
                          prepared_arrays: list[np.ndarray], device: str, batch_size: int = 16) -> dict:
    """Only image pixels enter this function; no target labels or fold metadata."""
    if not image_ids or len(image_ids) != len(native_arrays) or len(image_ids) != len(prepared_arrays):
        raise ValueError('Nonempty aligned input lists required')
    if any(a.dtype != np.uint8 or a.shape != (320, 320) for a in prepared_arrays):
        raise ValueError('Prepared 320×320 grayscale uint8 arrays required')
    morphology = np.asarray([list(_image_features(a).values()) for a in native_arrays], dtype=np.float32)
    images = torch.from_numpy(np.stack(prepared_arrays)[:, None].astype(np.float32) / 255)
    model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    global_arrays, spatial_arrays = [], []
    for start in range(0, len(images), batch_size):
        raw = images[start:start + batch_size].to(device).repeat(1, 3, 1, 1)
        feature_map = model((raw - mean) / std)
        global_arrays.append(F.adaptive_avg_pool2d(feature_map, 1).flatten(1).cpu().numpy())
        spatial_arrays.append(grid_pool(feature_map).cpu().numpy())
    if device == 'mps':
        torch.mps.synchronize()
    global_features = np.concatenate([np.concatenate(global_arrays), morphology], axis=1)
    return {
        'resnet18_morph': {'image_ids': np.array(image_ids), 'global_features': global_features,
                           'spatial_features': np.concatenate(spatial_arrays)},
    }
