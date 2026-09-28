"""V13 hip positioning/ROI pipeline, ported from skeleton-city (2).ipynb.

The original five outer folds, nested threshold selection, ResNet18 fine-tuning,
ROI metadata stack, and operating decisions are retained. Images must already be
prepared as 320x320 PNGs, with matching manifest.jsonl and splits.json.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import threading
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    AutoImageProcessor = AutoModel = None

DEFAULT_MODEL_ID = "microsoft/resnet-18"
MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = None             # Record the resolved HF commit in run metadata.
IMAGE_SIZE = 320                 # Native prepared resolution. Do not lower for this experiment.
SEED = 20260915
EPOCHS = 30                      # Upper bound; early stopping normally ends sooner.
FREEZE_EPOCHS = 5                # Train the new heads first, then fine-tune the backbone.
PATIENCE = 5                     # Consecutive non-improving validation epochs.
LR_HEAD = 2e-4
LR_BACKBONE = 7e-6
WEIGHT_DECAY = 3e-3
DROPOUT = 0.35
POS_WEIGHT_POWER = 0.5           # sqrt(class ratio): less false-positive pressure than full ratio.
POS_WEIGHT_MAX = 5.0
POSITIVE_LABEL_SMOOTHING = 0.02
HFLIP_PROB = 0.5                 # Preserved for the unchanged ROI branch.
P_CANONICALIZE_LATERALITY = True # Mirror right hips into one anatomical orientation.
P_TRAIN_HFLIP_PROB = 0.0         # Do not destroy canonical orientation during P training.
P_THRESHOLD_OBJECTIVE = "f1"     # Restored after V10 F0.5 reduced outer recall too strongly.
ROI_INFERENCE = "native"          # Accepted native neural member inference.
ROI_METADATA_C = 1.0               # Fixed before V13 outer evaluation.
ROI_STACK_C = 1.0                  # Unweighted second-level logistic model.
REFIT_AFTER_SELECTION = False    # Keep selection model and its probability scale for outer scoring.
ROI_TARGET_DRAW_FRACTION = 0.15  # Expected fraction; weights are balanced by study within class.
ROI_POS_WEIGHT_WITH_SAMPLER = False  # Avoid stacking ROI oversampling and positive loss weight.
INNER_CALIBRATION_ENSEMBLE = True   # Four leakage-free, separately calibrated inner members.
AMP_ENABLED = False               # FP32 is fast enough here and avoids T4 FP16 gradient overflow.
MAX_SKIPPED_BATCHES = 3
MIN_FINETUNE_EPOCHS = 5
MIN_RECALL = {"positioning_rotation": 0.70, "roi": 0.50}
BATCH_PER_GPU = 16
NUM_WORKERS = 0 if sys.platform == "darwin" else 2
OPERATING_THRESHOLDS = {"positioning_rotation": 0.2666015625, "roi": 0.408203125}


DATA_ROOT: Path | None = None
OUTPUT_DIR: Path | None = None
DEVICE = torch.device("cpu")
N_GPU = 0
BATCH_SIZE = 16
processor = None
image_mean = image_std = None
hf_kwargs = {}
MODEL_SOURCE_PROVENANCE = {}
_RUN_PROGRESS = ContextVar("hip_training_progress", default=None)
_TRAIN_LOCK = threading.RLock()


def _report(*values):
    """Keep historical trainer messages local to the explicitly selected run."""
    progress = _RUN_PROGRESS.get()
    if progress is None:
        print(*values)
    else:
        progress.update("training", " ".join(str(value) for value in values))


def _compatible_signature(previous, current):
    """Identical local model contents may be resumed after moving computers."""
    previous, current = dict(previous), dict(current)
    # The seed, data, epochs and actual backbone hashes remain mandatory.
    # Device and absolute snapshot paths are machine-specific provenance.
    for signature in (previous, current):
        signature.pop("device_type", None)
        provenance = dict(signature.get("model_source_provenance", {}))
        if provenance.get("source") == "local_snapshot":
            provenance.pop("path", None)
            provenance.pop("resolved_revision", None)
            signature.pop("model_id", None)
        signature["model_source_provenance"] = provenance
    return previous == current


def _model_source_provenance(model_id, processor):
    """Record the exact local model files (or resolved Hub revision)."""
    source_path = Path(model_id)
    if source_path.is_dir():
        file_hashes = {}
        for name in ("config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"):
            file_path = source_path / name
            if file_path.is_file():
                file_hashes[name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if "config.json" not in file_hashes or not ({"model.safetensors", "pytorch_model.bin"} & file_hashes.keys()):
            raise ValueError(f"HIP_MODEL_PATH is not a complete local ResNet18 snapshot: {source_path}")
        from .bootstrap import FILES as pinned_hashes, MODEL_REVISION as pinned_revision
        resolved_revision = pinned_revision if file_hashes == pinned_hashes else (
            source_path.name if len(source_path.name) == 40 and all(
                c in "0123456789abcdef" for c in source_path.name.lower()
            ) else None
        )
        return {
            "source": "local_snapshot",
            "path": str(source_path.resolve()),
            "resolved_revision": resolved_revision,
            "file_sha256": file_hashes,
        }
    return {
        "source": "huggingface_hub",
        "model_id": model_id,
        "resolved_revision": getattr(processor, "_commit_hash", None),
    }


def _configure(data_root, output_dir, device=None):
    """Bind data and accelerator for one sequential training/inference run."""
    global DATA_ROOT, OUTPUT_DIR, DEVICE, N_GPU, BATCH_SIZE
    global processor, image_mean, image_std, hf_kwargs
    global MODEL_ID, MODEL_SOURCE_PROVENANCE
    if AutoImageProcessor is None or AutoModel is None:
        raise ImportError(
            "Hip model needs transformers==4.49.0 and safetensors==0.5.2. "
            "Install the project requirements before training or model inference."
        )
    DATA_ROOT, OUTPUT_DIR = Path(data_root), Path(output_dir)
    override = os.environ.get("HIP_MODEL_PATH")
    if override and not Path(override).is_dir():
        raise FileNotFoundError(f"HIP_MODEL_PATH must name an existing local snapshot directory: {override}")
    from .bootstrap import ensure_pretrained
    active_progress = _RUN_PROGRESS.get()
    bundled_snapshot = OUTPUT_DIR / "pretrained/resnet18"
    if override:
        MODEL_ID = str(Path(override).resolve())
    else:
        bundled_snapshot = ensure_pretrained(bundled_snapshot, verbose=bool(active_progress and active_progress.enabled))
        MODEL_ID = str(bundled_snapshot.resolve())
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    DEVICE = torch.device(device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if DEVICE.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    N_GPU = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
    BATCH_SIZE = BATCH_PER_GPU * max(N_GPU, 1)
    hf_kwargs = {"local_files_only": True}
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False, **hf_kwargs)
    MODEL_SOURCE_PROVENANCE = _model_source_provenance(MODEL_ID, processor)
    image_mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(3, 1, 1)
    image_std = torch.tensor(processor.image_std, dtype=torch.float32).view(3, 1, 1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _report(f"Hip data: {DATA_ROOT} | output: {OUTPUT_DIR} | device: {DEVICE} | batch size: {BATCH_SIZE}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_records(data_root):
    """Validate the exact 150 labeled hip cases and study-disjoint outer folds."""
    DATA_ROOT = Path(data_root)
    for required in (DATA_ROOT / "images", DATA_ROOT / "manifest.jsonl", DATA_ROOT / "splits.json"):
        if not required.exists():
            raise FileNotFoundError(f"Missing required hip input: {required}")
    with (DATA_ROOT / "splits.json").open(encoding="utf-8") as handle:
        split_payload = json.load(handle)

    image_to_fold = {}
    fold_specs = split_payload["folds"]
    if isinstance(fold_specs, dict):
        # Current project schema: {"0": {"train": [...], "val": [...]}, ...}.
        fold_items = sorted(fold_specs.items(), key=lambda item: int(item[0]))
    elif isinstance(fold_specs, list):
        # Compatibility with an earlier list-based schema, retained for portable runs.
        fold_items = [(item.get("fold", index), item) for index, item in enumerate(fold_specs)]
    else:
        raise TypeError("splits.json 'folds' must be a mapping or a list")

    for fold_key, fold_spec in fold_items:
        fold_number = int(fold_key)
        val_image_ids = fold_spec.get("val", fold_spec.get("val_image_ids"))
        if val_image_ids is None:
            raise KeyError(f"Fold {fold_number} has neither 'val' nor 'val_image_ids'")
        for image_id in val_image_ids:
            if image_id in image_to_fold:
                raise ValueError(f"Duplicate validation image assignment: {image_id}")
            image_to_fold[image_id] = fold_number

    records = []
    with (DATA_ROOT / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if (
                item["anatomical_region"] not in {"right_hip", "left_hip"}
                or not item["eligible"]
                or item["source_split"] != "train"
            ):
                continue
            image_id = item["image_id"]
            if image_id not in image_to_fold:
                raise ValueError(f"Eligible hip missing from supplied split: {image_id}")
            labels = item["targets"]
            if any(labels.get(name) not in (0, 1) for name in ("hip_positioning_rotation", "hip_roi")):
                raise ValueError(f"Eligible hip has an unknown target: {image_id}")
            records.append({
                "image_id": image_id,
                "image_path": item["image_path"],
                "study_id": item["study_id"],
                "laterality": item["anatomical_region"],
                "fold": image_to_fold[image_id],
                "positioning_rotation": int(labels["hip_positioning_rotation"]),
                "roi": int(labels["hip_roi"]),
                "letterbox_size": int(item["letterbox_size"]),
                "resized_height": int(item["resized_height"]),
                "resized_width": int(item["resized_width"]),
                "pad_top": int(item["pad_top"]),
                "pad_bottom": int(item["pad_bottom"]),
                "pad_left": int(item["pad_left"]),
                "pad_right": int(item["pad_right"]),
                "original_height": int(item["original_height"]),
                "original_width": int(item["original_width"]),
            })

    records.sort(key=lambda row: row["image_id"])
    assert len(records) == 150, f"Expected 150 eligible hips, got {len(records)}"
    assert len({row["image_id"] for row in records}) == len(records)
    assert set(image_to_fold.values()) == set(range(5))
    manifest_digest = hashlib.sha256((DATA_ROOT / "manifest.jsonl").read_bytes()).hexdigest()
    assert manifest_digest == split_payload["manifest_sha256"], "Stale split/manifest pairing"
    hip_image_ids = {row["image_id"] for row in records}
    assert hip_image_ids.issubset(image_to_fold), "An eligible hip is missing from the supplied split"
    for fold in range(5):
        train_studies = {row["study_id"] for row in records if row["fold"] != fold}
        val_studies = {row["study_id"] for row in records if row["fold"] == fold}
        assert train_studies.isdisjoint(val_studies), f"Study leakage in fold {fold}"

    _report("Eligible hips:", len(records))
    for target in ("positioning_rotation", "roi"):
        _report(target, sum(row[target] for row in records), "/", len(records))
    _report("Fold sizes:", {fold: sum(row["fold"] == fold for row in records) for fold in range(5)})
    return records

GEOMETRY_FEATURE_NAMES = (
    "superior_joint_intensity_ratio",
    "joint_field_intensity_ratio",
    "inferior_shaft_intensity_ratio",
    "inferior_shaft_foreground_fraction",
    "lateral_bone_intensity_max_ratio",
    "lateral_intensity_asymmetry_ratio",
    "foreground_bbox_height_fraction",
    "foreground_bbox_width_fraction",
)
GEOMETRY_FEATURE_COUNT = len(GEOMETRY_FEATURE_NAMES)

def hip_geometry_features(pixels):
    '''Fixed-region intensity/coverage proxies; no anatomical localization is implied.'''
    h, w = pixels.shape
    whole = pixels[round(0.05 * h):round(0.95 * h), round(0.05 * w):round(0.95 * w)]
    # A hip image contains one proximal femur, unlike the bilateral lower-corner
    # zones in the original spine example. Use superior joint and central shaft zones.
    superior_joint = pixels[
        round(0.05 * h):round(0.35 * h), round(0.15 * w):round(0.85 * w)
    ]
    joint_field = pixels[
        round(0.18 * h):round(0.68 * h), round(0.08 * w):round(0.92 * w)
    ]
    inferior_shaft = pixels[
        round(0.65 * h):round(0.98 * h), round(0.20 * w):round(0.80 * w)
    ]
    lateral_left = pixels[
        round(0.30 * h):round(0.75 * h), round(0.02 * w):round(0.40 * w)
    ]
    lateral_right = pixels[
        round(0.30 * h):round(0.75 * h), round(0.60 * w):round(0.98 * w)
    ]
    denominator = float(whole.mean()) + 1e-6
    mask_threshold = max(0.02, float(np.quantile(pixels, 0.75)) * 0.10)
    foreground = pixels > mask_threshold
    ys, xs = np.where(foreground)
    if len(ys):
        bbox_h = (int(ys.max()) - int(ys.min()) + 1) / h
        bbox_w = (int(xs.max()) - int(xs.min()) + 1) / w
    else:
        bbox_h = bbox_w = 0.0
    shaft_foreground = foreground[
        round(0.65 * h):round(0.98 * h), round(0.20 * w):round(0.80 * w)
    ]
    lateral_left_ratio = float(lateral_left.mean()) / denominator
    lateral_right_ratio = float(lateral_right.mean()) / denominator
    features = np.asarray([
        superior_joint.mean() / denominator,
        joint_field.mean() / denominator,
        inferior_shaft.mean() / denominator,
        shaft_foreground.mean(),
        max(lateral_left_ratio, lateral_right_ratio),
        abs(lateral_left_ratio - lateral_right_ratio),
        bbox_h,
        bbox_w,
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=0.0).clip(0.0, 3.0)

class HipDataset(Dataset):
    def __init__(self, rows, *, training=False, canonicalize_laterality=False,
                 force_mirror=False, hflip_probability=HFLIP_PROB,
                 data_root=None, normalization_mean=None, normalization_std=None):
        self.rows = list(rows)
        self.training = training
        self.canonicalize_laterality = canonicalize_laterality
        self.force_mirror = force_mirror
        self.hflip_probability = float(hflip_probability)
        # Dataset instances are sent to spawned DataLoader workers on macOS.
        # Keep their inputs in the instance instead of relying on module globals.
        self.data_root = Path(data_root) if data_root is not None else DATA_ROOT
        self.normalization_mean = normalization_mean if normalization_mean is not None else image_mean
        self.normalization_std = normalization_std if normalization_std is not None else image_std
        if not 0.0 <= self.hflip_probability <= 1.0:
            raise ValueError("hflip_probability must be in [0, 1]")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["image_path"]
        with Image.open(path) as image:
            image = image.convert("L")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Unexpected prepared size {image.size} for {path.name}")
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        # Canonicalization is deterministic and occurs before any optional augmentation.
        # Labels are source labels; only the pixel coordinate system changes.
        if self.canonicalize_laterality and row["laterality"] == "right_hip":
            pixels = np.fliplr(pixels).copy()
        if self.force_mirror:
            pixels = np.fliplr(pixels).copy()
        if self.training and np.random.random() < self.hflip_probability:
            pixels = np.fliplr(pixels).copy()
        geometry = hip_geometry_features(pixels)
        if self.training:
            # Photometric augmentation only: geometry/rotation/coverage labels must not be altered.
            foreground = pixels > 0.01
            gamma = np.random.uniform(0.90, 1.10)
            gain = np.random.uniform(0.90, 1.10)
            noise = np.random.normal(0.0, np.random.uniform(0.0, 0.012), pixels.shape).astype(np.float32)
            augmented = np.power(np.clip(pixels, 0.0, 1.0), gamma) * gain
            pixels = np.where(foreground, augmented + noise, pixels).clip(0.0, 1.0).astype(np.float32)
        # Repetition, rather than synthetic colorization, preserves the original grayscale signal.
        tensor = torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)
        tensor = (tensor - self.normalization_mean) / self.normalization_std
        # -1 marks an unlabelled image during pure inference. Training rows have
        # both targets and retain the original notebook's target values.
        target = torch.tensor([
            row.get("positioning_rotation", -1), row.get("roi", -1),
        ], dtype=torch.float32)
        return tensor, torch.from_numpy(geometry), target, row["image_id"]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def study_balanced_roi_sampling_weights(rows, target_fraction):
    '''Expected ROI-positive draw fraction with equal mass per study inside each class.'''
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("ROI_TARGET_DRAW_FRACTION must be between zero and one")
    grouped = {0: {}, 1: {}}
    for index, row in enumerate(rows):
        label = int(row["roi"])
        grouped[label].setdefault(row["study_id"], []).append(index)
    if not grouped[0] or not grouped[1]:
        raise ValueError("ROI sampling requires both classes and at least one study per class")
    weights = np.zeros(len(rows), dtype=np.float64)
    for label, total_mass in ((0, 1.0 - target_fraction), (1, target_fraction)):
        study_mass = total_mass / len(grouped[label])
        for indices in grouped[label].values():
            weights[indices] = study_mass / len(indices)
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("ROI sampling weights do not sum to one")
    return weights

def make_loader(rows, *, shuffle, seed=SEED, canonicalize_laterality=False,
                force_mirror=False, hflip_probability=HFLIP_PROB):
    generator = torch.Generator().manual_seed(seed + (1 if shuffle else 0))
    sampler = None
    if shuffle and ROI_TARGET_DRAW_FRACTION is not None:
        weights = study_balanced_roi_sampling_weights(rows, ROI_TARGET_DRAW_FRACTION)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), num_samples=len(rows),
            replacement=True, generator=generator,
        )
    return DataLoader(
        HipDataset(
            rows, training=shuffle, canonicalize_laterality=canonicalize_laterality,
            force_mirror=force_mirror, hflip_probability=hflip_probability,
            data_root=DATA_ROOT, normalization_mean=image_mean, normalization_std=image_std,
        ), batch_size=BATCH_SIZE,
        shuffle=shuffle and sampler is None, sampler=sampler, generator=generator,
        worker_init_fn=seed_worker,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

# ---- Shared image encoder with corrected hip-geometry fusion. ----
class HipResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_ID, **hf_kwargs)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Dropout(p=DROPOUT),
        )
        fused_size = self.backbone.config.hidden_sizes[-1] + 16
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_size),
            nn.Dropout(p=DROPOUT),
            nn.Linear(fused_size, 128),
            nn.SiLU(),
            nn.Dropout(p=DROPOUT),
        )
        self.positioning_rotation_head = nn.Linear(128, 1)
        self.roi_head = nn.Linear(128, 1)

    def head_parameters(self):
        return (
            list(self.geometry_encoder.parameters())
            + list(self.fusion.parameters())
            + list(self.positioning_rotation_head.parameters())
            + list(self.roi_head.parameters())
        )

    def forward(self, pixel_values, geometry_features):
        outputs = self.backbone(pixel_values=pixel_values)
        # Hugging Face ResNet exposes a pooled [B, C, 1, 1] tensor; the
        # classification head requires one [B, C] vector per image.
        image_features = torch.flatten(outputs.pooler_output, start_dim=1)
        geometry_embedding = self.geometry_encoder(geometry_features)
        fused = self.fusion(torch.cat([image_features, geometry_embedding], dim=1))
        return torch.cat([
            self.positioning_rotation_head(fused), self.roi_head(fused),
        ], dim=1)

def build_model():
    model = HipResNet18().to(DEVICE)
    if DEVICE.type == "cuda" and N_GPU > 1:
        model = nn.DataParallel(model)
    return model

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def train_mode(model):
    model.train()
    # requires_grad=False alone does not freeze BatchNorm running statistics.
    # Small per-device batches must not overwrite pretrained normalization buffers.
    for module in unwrap(model).backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

def gradients_are_finite(model):
    return all(torch.isfinite(parameter.grad).all().item()
               for parameter in model.parameters() if parameter.grad is not None)

def evaluate(model, loader, criterion):
    model.eval()
    losses, probabilities, labels, image_ids = [], [], [], []
    count = 0
    with torch.no_grad():
        for pixels, geometry, targets, ids in loader:
            pixels = pixels.to(DEVICE, non_blocking=True)
            geometry = geometry.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED and DEVICE.type == "cuda"):
                logits = model(pixels, geometry)
                loss = criterion(logits, targets)
            losses.append(float(loss.detach().cpu()) * len(targets))
            count += len(targets)
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(targets.cpu().numpy())
            image_ids.extend(ids)
    return float(sum(losses) / count), np.concatenate(probabilities), np.concatenate(labels), image_ids

def metric_bundle(y_true, y_prob, thresholds=OPERATING_THRESHOLDS):
    result = {}
    for index, name in enumerate(("positioning_rotation", "roi")):
        truth = y_true[:, index].astype(int)
        probability = y_prob[:, index]
        threshold = float(thresholds[name])
        prediction = (probability >= threshold).astype(int)
        result[name] = {
            "n": int(len(truth)), "positives": int(truth.sum()), "threshold": threshold,
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "roc_auc": float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None,
            "average_precision": float(average_precision_score(truth, probability)) if truth.sum() else None,
            "tp": int(((truth == 1) & (prediction == 1)).sum()),
            "fp": int(((truth == 0) & (prediction == 1)).sum()),
            "fn": int(((truth == 1) & (prediction == 0)).sum()),
            "tn": int(((truth == 0) & (prediction == 0)).sum()),
        }
    return result

class PositiveSmoothedBCEWithLogitsLoss(nn.Module):
    '''Slightly soften positives without turning negatives into pseudo-positives.'''
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        smoothed_targets = targets * (1.0 - POSITIVE_LABEL_SMOOTHING)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed_targets, pos_weight=self.pos_weight,
        )

# ---- Inner selection -> optional refit -> outer evaluation. ----
def suggest_f1_thresholds(y_true, y_prob):
    suggestions = {}
    for index, name in enumerate(("positioning_rotation", "roi")):
        truth = y_true[:, index].astype(int)
        probability = y_prob[:, index]
        if len(np.unique(truth)) != 2 or not np.isfinite(probability).all():
            raise ValueError(f"Threshold selection requires both classes and finite scores: {name}")
        feasible = []
        for threshold in np.unique(np.concatenate(([0.0], probability, [1.0]))):
            prediction = (probability >= threshold).astype(int)
            recall = recall_score(truth, prediction, zero_division=0)
            if recall < MIN_RECALL[name]:
                continue
            precision = precision_score(truth, prediction, zero_division=0)
            f1 = f1_score(truth, prediction, zero_division=0)
            balanced = balanced_accuracy_score(truth, prediction)
            feasible.append((f1, balanced, precision, float(threshold), recall))
        f1, balanced, precision, threshold, recall = max(feasible)
        suggestions[name] = {
            "threshold": threshold, "minimum_recall_constraint": MIN_RECALL[name],
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "balanced_accuracy": float(balanced),
        }
    return suggestions

ROI_ACQUISITION_FEATURE_NAMES = (
    "content_height_fraction", "content_width_fraction",
    "pad_top_fraction", "pad_bottom_fraction",
    "pad_left_fraction", "pad_right_fraction", "source_aspect_ratio",
)

def roi_acquisition_features(rows):
    values = []
    for row in rows:
        size = float(row["letterbox_size"])
        width = max(float(row["original_width"]), 1.0)
        values.append([
            row["resized_height"] / size, row["resized_width"] / size,
            row["pad_top"] / size, row["pad_bottom"] / size,
            row["pad_left"] / size, row["pad_right"] / size,
            row["original_height"] / width,
        ])
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(ROI_ACQUISITION_FEATURE_NAMES):
        raise ValueError("Invalid ROI acquisition feature matrix")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite ROI acquisition features")
    return result

def build_roi_metadata_model(rows):
    labels = np.asarray([row["roi"] for row in rows], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("ROI metadata fit requires both classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=ROI_METADATA_C, class_weight="balanced", solver="liblinear",
            random_state=SEED, max_iter=1000,
        ),
    )
    return model.fit(roi_acquisition_features(rows), labels)

def build_roi_stacker(features, labels):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=ROI_STACK_C, class_weight=None, solver="liblinear",
            random_state=SEED, max_iter=1000,
        ),
    )
    return model.fit(np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=int))

def probability_logit(probability, eps=1e-6):
    probability = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    return np.log(probability / (1.0 - probability))

def suggest_binary_f1_threshold(truth, probability, minimum_recall):
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=np.float64)
    feasible = []
    for threshold in np.unique(np.concatenate(([0.0], probability, [1.0]))):
        prediction = (probability >= threshold).astype(int)
        recall = recall_score(truth, prediction, zero_division=0)
        if recall < minimum_recall:
            continue
        precision = precision_score(truth, prediction, zero_division=0)
        f1 = f1_score(truth, prediction, zero_division=0)
        balanced = balanced_accuracy_score(truth, prediction)
        feasible.append((f1, balanced, precision, float(threshold), recall))
    if not feasible:
        raise ValueError("No binary threshold satisfies the minimum-recall constraint")
    f1, balanced, precision, threshold, recall = max(feasible)
    return {
        "threshold": threshold, "minimum_recall_constraint": minimum_recall,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "balanced_accuracy": float(balanced),
    }

def sklearn_pipeline_payload(model):
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": logistic.coef_.tolist(),
        "intercept": logistic.intercept_.tolist(),
        "classes": logistic.classes_.tolist(),
    }

def paired_threshold_score(probability, threshold, eps=1e-6):
    '''Map a member's own threshold to 0.5 while retaining continuous distance.'''
    probability = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    threshold = np.clip(np.asarray(threshold, dtype=np.float64), eps, 1.0 - eps)
    centered_logit = (
        np.log(probability / (1.0 - probability))
        - np.log(threshold / (1.0 - threshold))
    )
    return 1.0 / (1.0 + np.exp(-centered_logit))

def fit_model(train_rows, *, seed, run_name, val_rows=None, epochs=None,
              canonicalize_laterality=False, hflip_probability=HFLIP_PROB):
    epochs = EPOCHS if epochs is None else epochs
    run_progress = _RUN_PROGRESS.get()
    if run_progress is not None:
        run_progress.update("training", f"Starting {run_name}", current=0, total=epochs)
    set_seed(seed)
    train_loader = make_loader(
        train_rows, shuffle=True, seed=seed,
        canonicalize_laterality=canonicalize_laterality,
        hflip_probability=hflip_probability,
    )
    val_loader = make_loader(
        val_rows, shuffle=False, seed=seed,
        canonicalize_laterality=canonicalize_laterality,
        hflip_probability=0.0,
    ) if val_rows is not None else None
    targets = np.array([[r["positioning_rotation"], r["roi"]] for r in train_rows], dtype=np.float32)
    positives = targets.sum(0)
    assert (positives > 0).all() and (positives < len(targets)).all()
    ratio = (len(targets) - positives) / positives
    pos_weight_values = np.clip(ratio ** POS_WEIGHT_POWER, 1.0, POS_WEIGHT_MAX)
    if ROI_TARGET_DRAW_FRACTION is not None and not ROI_POS_WEIGHT_WITH_SAMPLER:
        # The sampler is the ROI balancing mechanism in V6. Keeping the former
        # ROI pos_weight as well would double-count the same seven positives.
        pos_weight_values[1] = 1.0
    pos_weight = torch.tensor(pos_weight_values, dtype=torch.float32, device=DEVICE)
    criterion = PositiveSmoothedBCEWithLogitsLoss(pos_weight)
    model = build_model()
    core = unwrap(model)
    for parameter in core.backbone.parameters():
        parameter.requires_grad = False

    # One optimizer for the whole fit: preserve head moments when unfreezing.
    groups = []
    for parameters, base_lr in ((core.backbone.parameters(), LR_BACKBONE),
                                (core.head_parameters(), LR_HEAD)):
        parameters = list(parameters)
        for decay, selected in (
            (WEIGHT_DECAY, [p for p in parameters if p.ndim > 1]),
            (0.0, [p for p in parameters if p.ndim <= 1]),
        ):
            groups.append({"params": selected, "lr": base_lr, "base_lr": base_lr,
                           "weight_decay": decay})
    optimizer = torch.optim.AdamW(groups)
    scaler = GradScaler("cuda", enabled=AMP_ENABLED and DEVICE.type == "cuda")
    best_state, best_epoch, best_loss = None, 0, float("inf")
    stale, history = 0, []

    for epoch in range(epochs):
        if epoch == FREEZE_EPOCHS:
            for parameter in core.backbone.parameters():
                parameter.requires_grad = True
            stale = 0
        # Same schedule horizon in selection and refit, even if refit is shorter.
        if epoch < FREEZE_EPOCHS:
            factor = min(1.0, (epoch + 1) / 2.0)
        else:
            progress = (epoch - FREEZE_EPOCHS) / max(EPOCHS - FREEZE_EPOCHS - 1, 1)
            factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        train_mode(model)
        total_loss, count, skipped_batches = 0.0, 0, 0
        for pixels, geometry, truth, _ in train_loader:
            pixels, geometry, truth = (v.to(DEVICE, non_blocking=True) for v in (pixels, geometry, truth))
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED and DEVICE.type == "cuda"):
                logits = model(pixels, geometry)
                loss = criterion(logits, truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{run_name}: nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not gradients_are_finite(model):
                # Never let a NaN/Inf update corrupt a checkpoint. When AMP is
                # enabled, unscale_ has already recorded the overflow for update().
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                skipped_batches += 1
                if skipped_batches > MAX_SKIPPED_BATCHES:
                    raise RuntimeError(
                        f"{run_name}: more than {MAX_SKIPPED_BATCHES} nonfinite-gradient batches "
                        "within one epoch; inspect the inputs and reduce LR."
                    )
                continue
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                if skipped_batches > MAX_SKIPPED_BATCHES:
                    raise RuntimeError(f"{run_name}: repeated nonfinite gradient norms")
                continue
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(truth)
            count += len(truth)
        entry = {"epoch": epoch + 1, "train_loss": total_loss / count,
                 "head_lr": LR_HEAD * factor, "backbone_lr": LR_BACKBONE * factor,
                 "skipped_batches": skipped_batches}
        if val_loader is not None:
            val_loss, _, _, _ = evaluate(model, val_loader, criterion)
            entry["inner_val_loss"] = val_loss
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch + 1
        history.append(entry)
        if run_progress is None:
            _report(run_name, json.dumps(entry))
        else:
            run_progress.update(
                "epoch", f"{run_name} {json.dumps(entry)}", current=epoch + 1, total=epochs,
            )
        if val_loader is not None and epoch + 1 >= FREEZE_EPOCHS + MIN_FINETUNE_EPOCHS and stale >= PATIENCE:
            break
    if val_loader is not None:
        assert best_state is not None
        core.load_state_dict(best_state)
    return model, criterion, best_epoch, history


def train_evaluate(data_root, output_dir, device=None):
    """Train all 5 outer folds and save out-of-fold predictions and metrics.

    This is the original nested V13 evaluation, not a full-data deployment fit.
    Output includes fold checkpoints, OOF CSV, metrics JSON, and training histories.
    Completed folds can be resumed when dataset and protocol fingerprints match.
    """
    _configure(data_root, output_dir, device)
    set_seed(SEED)
    records = load_records(DATA_ROOT)
    run_signature = {
        "manifest_sha256": hashlib.sha256((DATA_ROOT / "manifest.jsonl").read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256((DATA_ROOT / "splits.json").read_bytes()).hexdigest(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_source_provenance": MODEL_SOURCE_PROVENANCE,
        "protocol": "V13_roi_metadata_stack",
        "seed": SEED,
        "epochs": EPOCHS,
        "device_type": DEVICE.type,
    }
    signature_path = OUTPUT_DIR / "run_signature.json"
    partial_path = OUTPUT_DIR / "oof_predictions_partial.json"
    if partial_path.exists():
        if not signature_path.exists() or not _compatible_signature(
            json.loads(signature_path.read_text(encoding="utf-8")), run_signature,
        ):
            raise ValueError("Existing hip partial results belong to another dataset/protocol; use a new output directory")
    else:
        signature_path.write_text(json.dumps(run_signature, indent=2), encoding="utf-8")
    partial_oof_path = OUTPUT_DIR / "oof_predictions_partial.json"
    if partial_oof_path.exists():
        all_oof = json.loads(partial_oof_path.read_text(encoding="utf-8"))
        if not isinstance(all_oof, list) or len({row.get("image_id") for row in all_oof}) != len(all_oof):
            raise ValueError(f"Invalid partial predictions: {partial_oof_path}")
        _report(f"Resuming from {len(all_oof)} completed predictions")
    else:
        all_oof = []

    fold_summaries = []
    by_id = {row["image_id"]: row for row in records}
    for fold in range(5):
        outer_rows = [row for row in records if row["fold"] == fold]
        history_path = OUTPUT_DIR / f"fold_{fold}_history.json"
        completed = [row for row in all_oof if row["fold"] == fold]
        if completed:
            if len(completed) != len(outer_rows) or not history_path.exists():
                raise RuntimeError(f"Incomplete persisted fold {fold}; use a new OUTPUT_DIR")
            fold_summaries.append(json.loads(history_path.read_text(encoding="utf-8")))
            missing = [path for path in _required_fold_checkpoints(OUTPUT_DIR, (fold,)) if not path.is_file()]
            if missing:
                raise RuntimeError(f"Persisted fold {fold} has missing checkpoints: {missing}")
            _report(f"fold={fold}: already complete; skipped")
            continue

        # Accepted V11 P branch: canonical training with inner-F1 threshold selection.
        p_inner_fold = (fold + 1) % 5
        p_fit_rows = [row for row in records if row["fold"] not in (fold, p_inner_fold)]
        p_inner_rows = [row for row in records if row["fold"] == p_inner_fold]
        p_model, p_criterion, p_epoch, p_history = fit_model(
            p_fit_rows, seed=SEED + fold, run_name=f"fold={fold}/v13_positioning_canonical_f1",
            val_rows=p_inner_rows, canonicalize_laterality=P_CANONICALIZE_LATERALITY,
            hflip_probability=P_TRAIN_HFLIP_PROB,
        )
        p_inner_loader = make_loader(
            p_inner_rows, shuffle=False, canonicalize_laterality=P_CANONICALIZE_LATERALITY,
            hflip_probability=0.0,
        )
        _, p_inner_prob, p_inner_truth, p_inner_ids = evaluate(
            p_model, p_inner_loader, p_criterion,
        )
        p_selection = suggest_f1_thresholds(p_inner_truth, p_inner_prob)["positioning_rotation"]
        p_selection["selection_metric"] = P_THRESHOLD_OBJECTIVE
        p_threshold = float(p_selection["threshold"])

        # Mirrored canonical scores are diagnostic only and never select the primary threshold.
        p_inner_mirror_loader = make_loader(
            p_inner_rows, shuffle=False, canonicalize_laterality=P_CANONICALIZE_LATERALITY,
            force_mirror=True, hflip_probability=0.0,
        )
        _, p_inner_prob_mirror, p_inner_truth_mirror, p_inner_ids_mirror = evaluate(
            p_model, p_inner_mirror_loader, p_criterion,
        )
        assert p_inner_ids == p_inner_ids_mirror and np.array_equal(p_inner_truth, p_inner_truth_mirror)

        p_outer_loader = make_loader(
            outer_rows, shuffle=False, canonicalize_laterality=P_CANONICALIZE_LATERALITY,
            hflip_probability=0.0,
        )
        _, p_outer_prob_all, p_outer_truth_all, p_outer_ids = evaluate(
            p_model, p_outer_loader, p_criterion,
        )
        p_outer_mirror_loader = make_loader(
            outer_rows, shuffle=False, canonicalize_laterality=P_CANONICALIZE_LATERALITY,
            force_mirror=True, hflip_probability=0.0,
        )
        _, p_outer_prob_mirror_all, p_outer_truth_mirror, p_outer_ids_mirror = evaluate(
            p_model, p_outer_mirror_loader, p_criterion,
        )
        assert p_outer_ids == p_outer_ids_mirror and np.array_equal(p_outer_truth_all, p_outer_truth_mirror)
        p_outer_prob = p_outer_prob_all[:, 0]
        p_outer_prob_mirror = p_outer_prob_mirror_all[:, 0]
        p_outer_truth = p_outer_truth_all[:, 0]
        p_core = unwrap(p_model)
        torch.save({
            "target": "positioning_rotation", "auxiliary_target": "roi", "fold": fold,
            "inner_fold": p_inner_fold, "selected_epochs": p_epoch,
            "threshold": p_threshold, "selection_metric": P_THRESHOLD_OBJECTIVE,
            "canonicalize_laterality": P_CANONICALIZE_LATERALITY,
            "training_hflip_probability": P_TRAIN_HFLIP_PROB,
            "training_protocol": "v13_unchanged_v12_positioning_branch",
            "fitted_image_ids": [row["image_id"] for row in p_fit_rows],
            "state_dict": {key: value.detach().cpu() for key, value in p_core.state_dict().items()},
        }, OUTPUT_DIR / f"fold_{fold}_positioning_v13.pt")
        del p_model, p_criterion, p_core
        gc.collect()
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        # V13 ROI: accepted native ensemble plus fully nested acquisition-geometry stack.
        roi_members, roi_summaries, roi_centered, roi_raw = [], [], [], []
        roi_stack_ids, roi_stack_targets = [], []
        roi_stack_deep, roi_stack_metadata, roi_stack_folds = [], [], []
        roi_outer_truth, roi_outer_ids = None, None
        for roi_inner_fold in [candidate for candidate in range(5) if candidate != fold]:
            roi_fit_rows = [row for row in records if row["fold"] not in (fold, roi_inner_fold)]
            roi_inner_rows = [row for row in records if row["fold"] == roi_inner_fold]
            roi_model, roi_criterion, roi_epoch, roi_history = fit_model(
                roi_fit_rows, seed=SEED + fold,
                run_name=f"fold={fold}/v13_roi_native_inner={roi_inner_fold}",
                val_rows=roi_inner_rows,
            )
            _, inner_prob_all, inner_truth_all, inner_ids = evaluate(
                roi_model, make_loader(roi_inner_rows, shuffle=False), roi_criterion,
            )
            selection_all = suggest_f1_thresholds(inner_truth_all, inner_prob_all)
            selection = selection_all["roi"]
            member_threshold = float(selection["threshold"])

            metadata_member = build_roi_metadata_model(roi_fit_rows)
            ordered_inner_rows = [by_id[image_id] for image_id in inner_ids]
            inner_metadata_probability = metadata_member.predict_proba(
                roi_acquisition_features(ordered_inner_rows)
            )[:, 1]
            inner_deep_centered = paired_threshold_score(
                inner_prob_all[:, 1], member_threshold,
            )
            roi_stack_ids.extend(inner_ids)
            roi_stack_targets.extend(inner_truth_all[:, 1].astype(int).tolist())
            roi_stack_deep.extend(inner_deep_centered.tolist())
            roi_stack_metadata.extend(inner_metadata_probability.tolist())
            roi_stack_folds.extend([roi_inner_fold] * len(inner_ids))

            _, outer_prob_all, outer_truth_all, outer_ids = evaluate(
                roi_model, make_loader(outer_rows, shuffle=False), roi_criterion,
            )
            outer_prob = outer_prob_all[:, 1]
            outer_truth = outer_truth_all[:, 1]
            if roi_outer_truth is None:
                roi_outer_truth, roi_outer_ids = outer_truth, outer_ids
            else:
                assert roi_outer_ids == outer_ids and np.array_equal(roi_outer_truth, outer_truth)
            roi_raw.append(outer_prob.astype(np.float64))
            roi_centered.append(paired_threshold_score(outer_prob, member_threshold))
            roi_core = unwrap(roi_model)
            roi_members.append({
                "inner_fold": roi_inner_fold, "selected_epochs": roi_epoch,
                "threshold": member_threshold, "inference": ROI_INFERENCE,
                "fitted_image_ids": [row["image_id"] for row in roi_fit_rows],
                "state_dict": {key: value.detach().cpu() for key, value in roi_core.state_dict().items()},
            })
            roi_summaries.append({
                "inner_fold": roi_inner_fold, "selected_epochs": roi_epoch,
                "selection": selection, "fit_ids": [row["image_id"] for row in roi_fit_rows],
                "inner_ids": inner_ids, "selection_history": roi_history,
                "inner_probabilities": inner_prob_all[:, 1].tolist(),
                "inner_metadata_probabilities": inner_metadata_probability.tolist(),
                "inner_targets": inner_truth_all[:, 1].tolist(),
            })
            del roi_model, roi_criterion, roi_core
            gc.collect()
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        assert len(roi_stack_ids) == len(set(roi_stack_ids)) == 120
        assert set(roi_stack_ids) == {
            row["image_id"] for row in records if row["fold"] != fold
        }
        roi_stack_targets = np.asarray(roi_stack_targets, dtype=int)
        roi_stack_folds = np.asarray(roi_stack_folds, dtype=int)
        roi_stack_features = np.column_stack([
            probability_logit(roi_stack_deep),
            probability_logit(roi_stack_metadata),
        ])

        # Second-level cross-fitting: threshold never sees predictions from a stacker
        # trained on the same case.
        roi_stack_oof = np.zeros(len(roi_stack_targets), dtype=np.float64)
        for calibration_fold in sorted(set(roi_stack_folds.tolist())):
            calibration_val = roi_stack_folds == calibration_fold
            calibration_fit = ~calibration_val
            calibration_model = build_roi_stacker(
                roi_stack_features[calibration_fit], roi_stack_targets[calibration_fit],
            )
            roi_stack_oof[calibration_val] = calibration_model.predict_proba(
                roi_stack_features[calibration_val]
            )[:, 1]
        roi_stack_selection = suggest_binary_f1_threshold(
            roi_stack_targets, roi_stack_oof, MIN_RECALL["roi"],
        )
        roi_stack_threshold = float(roi_stack_selection["threshold"])
        roi_stacker = build_roi_stacker(roi_stack_features, roi_stack_targets)

        roi_outer_prob_deep = np.mean(roi_centered, axis=0)
        roi_raw_mean = np.mean(roi_raw, axis=0)
        outer_train_rows = [row for row in records if row["fold"] != fold]
        roi_metadata_model = build_roi_metadata_model(outer_train_rows)
        ordered_outer_rows = [by_id[image_id] for image_id in roi_outer_ids]
        roi_outer_prob_metadata = roi_metadata_model.predict_proba(
            roi_acquisition_features(ordered_outer_rows)
        )[:, 1]
        roi_outer_stack_features = np.column_stack([
            probability_logit(roi_outer_prob_deep),
            probability_logit(roi_outer_prob_metadata),
        ])
        roi_outer_prob = roi_stacker.predict_proba(roi_outer_stack_features)[:, 1]

        torch.save({
            "target": "roi", "auxiliary_target": "positioning_rotation", "fold": fold,
            "training_protocol": "v13_native_neural_members_plus_nested_metadata_stack",
            "inference": ROI_INFERENCE, "members": roi_members,
        }, OUTPUT_DIR / f"fold_{fold}_roi_v13_ensemble.pt")
        calibration_payload = {
            "fold": fold, "threshold": roi_stack_threshold,
            "selection": roi_stack_selection,
            "feature_order": ["centered_neural_logit", "acquisition_metadata_logit"],
            "acquisition_feature_names": list(ROI_ACQUISITION_FEATURE_NAMES),
            "metadata_model": sklearn_pipeline_payload(roi_metadata_model),
            "stacker_model": sklearn_pipeline_payload(roi_stacker),
        }
        (OUTPUT_DIR / f"fold_{fold}_roi_v13_calibration.json").write_text(
            json.dumps(calibration_payload, separators=(",", ":")), encoding="utf-8",
        )

        assert p_outer_ids == roi_outer_ids
        summary = {
            "fold": fold, "training_protocol": "v13_v12_positioning_plus_nested_roi_metadata_stack",
            "positioning_rotation": {
                "inner_fold": p_inner_fold, "selected_epochs": p_epoch,
                "threshold": p_threshold, "selection": p_selection,
                "fit_ids": [row["image_id"] for row in p_fit_rows],
                "inner_ids": p_inner_ids, "selection_history": p_history,
                "inner_probabilities": p_inner_prob[:, 0].tolist(),
                "inner_mirrored_probabilities": p_inner_prob_mirror[:, 0].tolist(),
                "inner_targets": p_inner_truth[:, 0].tolist(),
            },
            "roi": {
                "threshold": roi_stack_threshold, "selection": roi_stack_selection,
                "members": roi_summaries, "stack_ids": roi_stack_ids,
                "stack_targets": roi_stack_targets.tolist(),
                "stack_deep_probabilities": list(map(float, roi_stack_deep)),
                "stack_metadata_probabilities": list(map(float, roi_stack_metadata)),
                "stack_oof_probabilities": roi_stack_oof.tolist(),
            },
            "outer_ids": p_outer_ids,
        }
        fold_summaries.append(summary)
        history_path.write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
        for index, image_id in enumerate(p_outer_ids):
            p_tta_mean = 0.5 * (float(p_outer_prob[index]) + float(p_outer_prob_mirror[index]))
            row = {"image_id": image_id, "study_id": by_id[image_id]["study_id"],
                   "laterality": by_id[image_id]["laterality"], "fold": fold,
                   "positioning_rotation_true": int(p_outer_truth[index]),
                   "positioning_rotation_probability": float(p_outer_prob[index]),
                   "positioning_rotation_probability_mirrored": float(p_outer_prob_mirror[index]),
                   "positioning_rotation_probability_tta_mean_diagnostic": p_tta_mean,
                   "positioning_rotation_threshold": p_threshold,
                   "positioning_rotation_prediction": int(p_outer_prob[index] >= p_threshold),
                   "roi_true": int(roi_outer_truth[index]),
                   "roi_probability": float(roi_outer_prob[index]),
                   "roi_probability_deep_diagnostic": float(roi_outer_prob_deep[index]),
                   "roi_probability_metadata_diagnostic": float(roi_outer_prob_metadata[index]),
                   "roi_raw_probability_mean": float(roi_raw_mean[index]),
                   "roi_threshold": roi_stack_threshold,
                   "roi_prediction": int(roi_outer_prob[index] >= roi_stack_threshold)}
            all_oof.append(row)
        partial_oof_path.write_text(json.dumps(all_oof, separators=(",", ":")), encoding="utf-8")
        del roi_members
        gc.collect()
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    all_oof.sort(key=lambda row: row["image_id"])
    assert len(all_oof) == len(records) == len({row["image_id"] for row in all_oof})
    partial_oof_path.write_text(json.dumps(all_oof, separators=(",", ":")), encoding="utf-8")
    names = ("positioning_rotation", "roi")
    oof_truth = np.asarray([[row[f"{name}_true"] for name in names] for row in all_oof])
    oof_prob = np.asarray([[row[f"{name}_probability"] for name in names] for row in all_oof])
    oof_predictions = np.asarray([[row[f"{name}_prediction"] for name in names] for row in all_oof])
    overall_metrics = metric_bundle(oof_truth, oof_predictions, dict.fromkeys(names, 0.5))
    ranking = metric_bundle(oof_truth, oof_prob)
    for name in names:
        overall_metrics[name]["threshold"] = "target-specific inner protocol"
        overall_metrics[name]["roc_auc"] = ranking[name]["roc_auc"]
        overall_metrics[name]["average_precision"] = ranking[name]["average_precision"]
    _report("OUTER FOLDS: V13 unchanged V12 P + nested ROI acquisition stack")
    _report(json.dumps(overall_metrics, indent=2))
    historical_threshold_metrics = {"note": "Not comparable across raw P and centered ROI scores"}
    exploratory_threshold_suggestions = suggest_f1_thresholds(oof_truth, oof_prob)
    _report("EXPLORATORY ONLY: global OOF F1 thresholds, biased by selection")
    _report(json.dumps(exploratory_threshold_suggestions, indent=2))

    with (OUTPUT_DIR / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_oof[0]))
        writer.writeheader()
        writer.writerows(all_oof)

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": MODEL_SOURCE_PROVENANCE.get("resolved_revision"),
        "model_source_provenance": MODEL_SOURCE_PROVENANCE,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "epochs": EPOCHS,
        "freeze_epochs": FREEZE_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "lr_head": LR_HEAD,
        "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "pos_weight_power": POS_WEIGHT_POWER,
        "pos_weight_max": POS_WEIGHT_MAX,
        "positive_label_smoothing": POSITIVE_LABEL_SMOOTHING,
        "operating_thresholds": OPERATING_THRESHOLDS,
        "horizontal_flip_probability_roi": HFLIP_PROB,
        "positioning_canonicalize_laterality": P_CANONICALIZE_LATERALITY,
        "positioning_horizontal_flip_probability": P_TRAIN_HFLIP_PROB,
        "positioning_threshold_objective": P_THRESHOLD_OBJECTIVE,
        "roi_inference": ROI_INFERENCE,
        "roi_metadata_c": ROI_METADATA_C,
        "roi_stack_c": ROI_STACK_C,
        "roi_acquisition_feature_names": ROI_ACQUISITION_FEATURE_NAMES,
        "refit_after_selection": REFIT_AFTER_SELECTION,
        "roi_target_draw_fraction": ROI_TARGET_DRAW_FRACTION,
        "roi_pos_weight_with_sampler": ROI_POS_WEIGHT_WITH_SAMPLER,
        "inner_calibration_ensemble": INNER_CALIBRATION_ENSEMBLE,
        "batchnorm_running_statistics": "frozen",
        "evaluation_protocol": (
            "rotating inner fold selects epochs/thresholds; full outer-train refit; outer scored once"
            if REFIT_AFTER_SELECTION else
            "V13 held-out outer folds with inner epoch/threshold selection and cross-fitted ROI stack; "
            "inner validation labels also select member epochs and thresholds"
        ),
        "minimum_recall_constraints": MIN_RECALL,
        "architecture_note": "V13 V12 P plus V9 native ROI ensemble stacked with manifest acquisition geometry",
        "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
        "folds": 5,
        "batch_per_gpu": BATCH_PER_GPU,
        "visible_gpus": N_GPU,
        "metrics": overall_metrics,
        "historical_threshold_metrics": historical_threshold_metrics,
        "exploratory_threshold_suggestions": exploratory_threshold_suggestions,
        "fold_history_files": [f"fold_{fold}_history.json" for fold in range(5)],
        "input_manifest_sha256": sha256(DATA_ROOT / "manifest.jsonl"),
        "input_splits_sha256": sha256(DATA_ROOT / "splits.json"),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "sklearn": sklearn.__version__,
    }
    with (OUTPUT_DIR / "metrics_and_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    import shutil
    _report("Saved:", OUTPUT_DIR)
    _report("Interpretation caution: 7 ROI positives and 36 positioning/rotation positives make fold-level estimates unstable.")
    metadata_path = OUTPUT_DIR / "metrics_and_run_metadata.json"
    return {
        "metrics": overall_metrics,
        "predictions": all_oof,
        "metrics_path": metadata_path,
        "predictions_path": OUTPUT_DIR / "oof_predictions.csv",
        "output_dir": OUTPUT_DIR,
    }


def load_saved_predictions(data_root, output_dir, *, require_complete=True):
    """Load audited OOF scores without re-running the expensive 25-model fit.

    The supplied manifest, splits, labels and per-image thresholds are checked
    against the predictions; this function does not need transformers or a GPU.
    """
    data_root, output_dir = Path(data_root), Path(output_dir)
    manifest_path = data_root / "manifest.jsonl"
    split_path = data_root / "splits.json"
    pred_path = output_dir / "oof_predictions.csv"
    if not pred_path.exists():
        pred_path = output_dir / "oof_predictions_partial.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"No hip OOF predictions in {output_dir}")
    source = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("anatomical_region") in {"left_hip", "right_hip"}
                and row.get("eligible") and row.get("source_split") == "train"):
            image_id = row["image_id"]
            if image_id in source:
                raise ValueError(f"Duplicate hip image_id in manifest: {image_id}")
            source[image_id] = row
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if split_payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("Hip split/manifest hash mismatch")
    folds = split_payload["folds"]
    fold_items = folds.items() if isinstance(folds, dict) else (
        (item.get("fold", i), item) for i, item in enumerate(folds)
    )
    fold_by_id = {}
    for fold_id, spec in fold_items:
        for image_id in spec.get("val", spec.get("val_image_ids", [])):
            if image_id in fold_by_id:
                raise ValueError(f"Duplicate hip validation assignment: {image_id}")
            fold_by_id[image_id] = int(fold_id)
    metadata_path = output_dir / "metrics_and_run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("input_manifest_sha256")
        if expected and expected != manifest_sha:
            raise ValueError("Hip prediction manifest differs from training data")
        expected = metadata.get("input_splits_sha256")
        if expected and expected != hashlib.sha256(split_path.read_bytes()).hexdigest():
            raise ValueError("Hip prediction splits differ from training data")
    if pred_path.suffix == ".csv":
        with pred_path.open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
    else:
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    result = []
    seen = set()
    for raw in predictions:
        image_id = raw["image_id"]
        if image_id in seen or image_id not in source:
            raise ValueError(f"Duplicate/ineligible hip OOF image: {image_id}")
        seen.add(image_id)
        truth = source[image_id]
        row = dict(raw)
        row["fold"] = int(row["fold"])
        if row["fold"] != fold_by_id.get(image_id):
            raise ValueError(f"Incorrect hip OOF fold: {image_id}")
        if row.get("study_id", truth["study_id"]) != truth["study_id"]:
            raise ValueError(f"Study mismatch for hip OOF image: {image_id}")
        if row.get("laterality", truth["anatomical_region"]) != truth["anatomical_region"]:
            raise ValueError(f"Laterality mismatch for hip OOF image: {image_id}")
        for task in ("positioning_rotation", "roi"):
            label = int(row[f"{task}_true"])
            score = float(row[f"{task}_probability"])
            threshold = float(row[f"{task}_threshold"])
            prediction = int(row[f"{task}_prediction"])
            if label != truth["targets"][f"hip_{task}"]:
                raise ValueError(f"Hip label mismatch: {image_id}/{task}")
            if not (np.isfinite(score) and np.isfinite(threshold)
                    and 0 <= score <= 1 and 0 <= threshold <= 1):
                raise ValueError(f"Invalid hip probability/threshold: {image_id}/{task}")
            if prediction != int(score >= threshold):
                raise ValueError(f"Hip prediction disagrees with threshold: {image_id}/{task}")
            for suffix, value in (("true", label), ("probability", score),
                                  ("threshold", threshold), ("prediction", prediction)):
                row[f"{task}_{suffix}"] = value
        result.append(row)
    if require_complete and set(source) != seen:
        raise ValueError(f"Incomplete hip OOF predictions: {len(seen)}/{len(source)}")
    return sorted(result, key=lambda item: item["image_id"])


def _predict_probabilities(model, rows, *, canonicalize_laterality=False):
    loader = make_loader(
        rows, shuffle=False, canonicalize_laterality=canonicalize_laterality,
        hflip_probability=0.0,
    )
    model.eval()
    batches, ids = [], []
    with torch.no_grad():
        for pixels, geometry, _, batch_ids in loader:
            logits = model(pixels.to(DEVICE), geometry.to(DEVICE))
            batches.append(torch.sigmoid(logits.float()).cpu().numpy())
            ids.extend(batch_ids)
    if ids != [row["image_id"] for row in rows]:
        raise RuntimeError("Hip inference loader changed image order")
    probabilities = np.concatenate(batches) if batches else np.empty((0, 2), dtype=np.float32)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Hip neural model returned nonfinite/out-of-range probabilities")
    return probabilities


def _payload_probability(payload, features):
    features = np.asarray(features, dtype=np.float64)
    mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
    coef = np.asarray(payload["coef"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)
    if list(payload["classes"]) != [0, 1]:
        raise ValueError("Unsupported hip calibration class order")
    if (features.ndim != 2 or mean.shape != (features.shape[1],)
            or scale.shape != mean.shape or coef.shape != (1, features.shape[1])
            or intercept.shape != (1,) or not all(
                np.isfinite(array).all() for array in (features, mean, scale, coef, intercept)
            ) or (scale <= 0).any()):
        raise ValueError("Hip calibration has invalid dimensions or nonfinite parameters/features")
    logits = ((features - mean) / scale) @ coef[0] + intercept[0]
    # Stable sigmoid with the same logistic mapping as sklearn.
    probability = np.exp(-np.logaddexp(0.0, -logits))
    if not np.isfinite(probability).all():
        raise ValueError("Hip calibration returned nonfinite probabilities")
    return probability


def _model_from_state(state_dict):
    model = build_model()
    unwrap(model).load_state_dict(state_dict)
    model.eval()
    return model


def _score_one_fold(rows, fold, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    p_path = checkpoint_dir / f"fold_{fold}_positioning_v13.pt"
    roi_path = checkpoint_dir / f"fold_{fold}_roi_v13_ensemble.pt"
    cal_path = checkpoint_dir / f"fold_{fold}_roi_v13_calibration.json"
    for path in (p_path, roi_path, cal_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing trained hip checkpoint: {path}")
    p_checkpoint = torch.load(p_path, map_location="cpu", weights_only=True)
    p_model = _model_from_state(p_checkpoint["state_dict"])
    p_prob = _predict_probabilities(
        p_model, rows, canonicalize_laterality=p_checkpoint["canonicalize_laterality"],
    )[:, 0]
    p_threshold = float(p_checkpoint["threshold"])
    del p_model, p_checkpoint
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    roi_checkpoint = torch.load(roi_path, map_location="cpu", weights_only=True)
    centered_members = []
    for member in roi_checkpoint["members"]:
        model = _model_from_state(member["state_dict"])
        raw = _predict_probabilities(model, rows)[:, 1]
        centered_members.append(paired_threshold_score(raw, member["threshold"]))
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    del roi_checkpoint
    neural = np.mean(centered_members, axis=0)
    calibration = json.loads(cal_path.read_text(encoding="utf-8"))
    metadata_prob = _payload_probability(
        calibration["metadata_model"], roi_acquisition_features(rows)
    )
    stack_features = np.column_stack([
        probability_logit(neural), probability_logit(metadata_prob),
    ])
    roi_prob = _payload_probability(calibration["stacker_model"], stack_features)
    return {
        "positioning_rotation_probability": p_prob,
        "positioning_rotation_threshold": p_threshold,
        "roi_probability": roi_prob,
        "roi_threshold": float(calibration["threshold"]),
    }


def predict_prepared(data_root, checkpoint_dir, rows, device=None):
    """Score prepared 320x320 hip rows using saved V13 fold checkpoints.

    A row with ``fold`` uses precisely that held-out outer model and reproduces
    OOF inference. A row without ``fold`` receives the average of five fold
    scores centered on their own thresholds; this deployment aggregation has
    not been independently validated and is identified in the output.
    """
    rows = [dict(row) for row in rows]
    if not rows:
        return []
    from .preprocessing import validate_geometry
    for row in rows:
        row.setdefault("laterality", row.get("anatomical_region"))
        if row["laterality"] not in {"left_hip", "right_hip"}:
            raise ValueError(f"Missing/invalid hip laterality: {row.get('image_id')}")
        for name in ("image_id", "image_path", "letterbox_size", "resized_height",
                     "resized_width", "pad_top", "pad_bottom", "pad_left",
                     "pad_right", "original_height", "original_width"):
            if name not in row:
                raise KeyError(f"Hip prepared row missing {name}: {row.get('image_id')}")
        validate_geometry(row)
    _configure(data_root, checkpoint_dir, device)
    fold_rows = {fold: [] for fold in range(5)}
    for row in rows:
        if row.get("fold") is None:
            for fold in range(5):
                fold_rows[fold].append(row)
        else:
            fold = int(row["fold"])
            if fold not in fold_rows:
                raise ValueError(f"Invalid hip fold {fold} for {row['image_id']}")
            fold_rows[fold].append(row)
    scores_by_id = {row["image_id"]: [] for row in rows}
    if len(scores_by_id) != len(rows):
        raise ValueError("Duplicate hip image IDs")
    for fold, subset in fold_rows.items():
        if not subset:
            continue
        fold_scores = _score_one_fold(subset, fold, checkpoint_dir)
        if any(not np.isfinite(value).all() or np.any(np.asarray(value) < 0)
               or np.any(np.asarray(value) > 1) for value in fold_scores.values()):
            raise ValueError(f"Hip fold {fold} returned nonfinite/out-of-range probabilities or thresholds")
        for index, row in enumerate(subset):
            scores_by_id[row["image_id"]].append({
                name: float(value[index]) if hasattr(value, "__len__") else float(value)
                for name, value in fold_scores.items()
            })
    results = []
    for row in rows:
        members = scores_by_id[row["image_id"]]
        result = {"image_id": row["image_id"], "study_id": row.get("study_id"),
                  "laterality": row["laterality"], "fold": row.get("fold")}
        if len(members) == 1:
            result.update(members[0])
            result["score_type"] = "held_out_fold"
        else:
            for task in ("positioning_rotation", "roi"):
                centered = [paired_threshold_score(
                    member[f"{task}_probability"], member[f"{task}_threshold"]
                ) for member in members]
                result[f"{task}_probability"] = float(np.mean(centered))
                result[f"{task}_threshold"] = 0.5
            result["score_type"] = "five_fold_threshold_centered_mean"
        for task in ("positioning_rotation", "roi"):
            result[f"{task}_prediction"] = int(
                result[f"{task}_probability"] >= result[f"{task}_threshold"]
            )
            if task in row and row[task] in (0, 1):
                result[f"{task}_true"] = int(row[task])
        results.append(result)
    return results


def _resolve_one_prepared_record(data_root, image):
    """Accept a manifest row, image ID, or prepared PNG path."""
    if isinstance(image, Mapping):
        return dict(image)
    wanted = str(image)
    matches = []
    manifest_path = Path(data_root) / "manifest.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("anatomical_region") not in {"left_hip", "right_hip"}:
            continue
        path = Path(data_root) / row["image_path"]
        if wanted in {row["image_id"], row["image_path"], path.name} or Path(wanted).resolve() == path.resolve():
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one prepared hip image for {wanted!r}, found {len(matches)}")
    return matches[0]


def _required_fold_checkpoints(checkpoint_dir, folds):
    checkpoint_dir = Path(checkpoint_dir)
    return [
        checkpoint_dir / f"fold_{fold}_{suffix}"
        for fold in folds
        for suffix in (
            "positioning_v13.pt", "roi_v13_ensemble.pt",
            "roi_v13_calibration.json",
        )
    ]


def describe_prepared_image(data_root, row):
    """Return image path and label-independent display geometry for one hip."""
    data_root = Path(data_root)
    row = dict(row)
    row.setdefault("laterality", row.get("anatomical_region"))
    if row["laterality"] not in {"left_hip", "right_hip"}:
        raise ValueError(f"Missing/invalid hip laterality: {row.get('image_id')}")
    image_path = (data_root / row["image_path"]).resolve()
    images_root = (data_root / "images").resolve()
    if not image_path.is_relative_to(images_root):
        raise ValueError(f"Prepared hip image lies outside images/: {image_path}")
    with Image.open(image_path) as source_image:
        if source_image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} prepared hip PNG: {image_path}")
        pixels = np.asarray(source_image.convert("L"), dtype=np.float32) / 255.0
    size = int(row["letterbox_size"])
    if size != IMAGE_SIZE:
        raise ValueError(f"Letterbox metadata size {size} differs from prepared image {IMAGE_SIZE}")
    if (int(row["pad_top"]) + int(row["resized_height"]) + int(row["pad_bottom"]) != size
            or int(row["pad_left"]) + int(row["resized_width"]) + int(row["pad_right"]) != size):
        raise ValueError(f"Inconsistent letterbox metadata: {row['image_id']}")
    content_bbox = {
        "x0": int(row["pad_left"]), "y0": int(row["pad_top"]),
        "x1": int(row["pad_left"]) + int(row["resized_width"]),
        "y1": int(row["pad_top"]) + int(row["resized_height"]),
    }
    foreground_threshold = max(0.02, float(np.quantile(pixels, 0.75)) * 0.10)
    foreground = pixels > foreground_threshold
    ys, xs = np.where(foreground)
    foreground_bbox = (
        {"x0": int(xs.min()), "y0": int(ys.min()),
         "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1}
        if len(xs) else None
    )
    roi_pixel_features = hip_geometry_features(pixels)
    positioning_pixels = (
        np.fliplr(pixels).copy()
        if P_CANONICALIZE_LATERALITY and row["laterality"] == "right_hip"
        else pixels
    )
    positioning_pixel_features = hip_geometry_features(positioning_pixels)
    acquisition = roi_acquisition_features([row])[0]
    geometry = {
        "coordinate_system": "prepared_image_pixels; top-left origin; x1/y1 exclusive",
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE,
        "letterbox_content_bbox": content_bbox,
        "foreground_bbox": foreground_bbox,
        "foreground_threshold_0_to_1": foreground_threshold,
        "foreground_fraction": float(foreground.mean()),
        "roi_pixel_geometry_features": dict(zip(
            GEOMETRY_FEATURE_NAMES, map(float, roi_pixel_features),
        )),
        "positioning_pixel_geometry_features": dict(zip(
            GEOMETRY_FEATURE_NAMES, map(float, positioning_pixel_features),
        )),
        "acquisition_features": dict(zip(
            ROI_ACQUISITION_FEATURE_NAMES, map(float, acquisition),
        )),
        "interpretation": (
            "The boxes show letterbox content and thresholded image foreground. "
            "They are not anatomical landmarks or a predicted ROI segmentation."
        ),
    }
    return image_path, geometry


def format_inference_result(row, image_path, raw, geometry):
    """Build the public one-image result from a batch model prediction."""
    raw = dict(raw)
    raw.pop("positioning_rotation_true", None)
    raw.pop("roi_true", None)
    raw["image_path"] = str(image_path)
    criteria = {
        "hip_positioning_rotation": {
            "score": raw["positioning_rotation_probability"],
            "threshold": raw["positioning_rotation_threshold"],
            "label": bool(raw["positioning_rotation_prediction"]),
        },
        "hip_roi": {
            "score": raw["roi_probability"],
            "threshold": raw["roi_threshold"],
            "label": bool(raw["roi_prediction"]),
        },
    }
    return {
        "image_id": raw["image_id"],
        "study_id": raw["study_id"],
        "anatomical_region": row.get("laterality", row.get("anatomical_region")),
        "labels": {task: item["label"] for task, item in criteria.items()},
        "scores": {task: item["score"] for task, item in criteria.items()},
        "geometry": geometry,
        "criteria": criteria,
        "raw": raw,
    }


def infer(data_root, checkpoint_dir, image, *, fold=None, device=None):
    """Infer both hip checks for one prepared image and return display data.

    ``image`` can be a manifest row, image ID, or prepared PNG path. A specified
    fold uses that fold's three saved artifacts. Without a fold, all five outer
    fold artifacts are required; the model averages threshold-centered scores.
    Coordinates are pixels in the original prepared 320x320 frame, with
    half-open x1/y1 bounds. The boxes are image-processing diagnostics and are
    neither anatomical landmarks nor a model-predicted ROI segmentation.
    """
    data_root, checkpoint_dir = Path(data_root), Path(checkpoint_dir)
    row = _resolve_one_prepared_record(data_root, image)
    row.setdefault("laterality", row.get("anatomical_region"))
    if fold is not None:
        row["fold"] = int(fold)
    selected_fold = row.get("fold")
    if selected_fold is not None and int(selected_fold) not in range(5):
        raise ValueError("Hip inference fold must be 0, 1, 2, 3, or 4")
    folds = range(5) if selected_fold is None else (int(selected_fold),)
    missing = [path for path in _required_fold_checkpoints(checkpoint_dir, folds)
               if not path.is_file()]
    if missing:
        need = "all five folds" if selected_fold is None else f"fold {selected_fold}"
        raise FileNotFoundError(
            f"Hip inference requires complete checkpoints for {need}; "
            f"missing: {', '.join(str(path) for path in missing)}. "
            "For an available single fold, pass fold=0..4 explicitly."
        )

    image_path, geometry = describe_prepared_image(data_root, row)
    # The public one-image API never returns or requires source labels.
    model_row = {key: value for key, value in row.items()
                 if key not in {"positioning_rotation", "roi", "targets"}}
    raw = predict_prepared(data_root, checkpoint_dir, [model_row], device=device)[0]
    return format_inference_result(row, image_path, raw, geometry)


def _source_artifact_paths(source_dir):
    return [
        *_required_fold_checkpoints(source_dir, range(5)),
        *(source_dir / f"fold_{fold}_history.json" for fold in range(5)),
        *(source_dir / name for name in (
            "run_signature.json", "oof_predictions_partial.json", "oof_predictions.csv",
            "metrics_and_run_metadata.json", "checkpoint_manifest.json",
        )),
    ]


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_complete_run(data_root, source_dir, epochs):
    """Return saved results only when all folds and the requested protocol agree."""
    required = _source_artifact_paths(source_dir)
    required = [path for path in required if path.name not in {
        "checkpoint_manifest.json", "oof_predictions_partial.json",
    }]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    signature = json.loads((source_dir / "run_signature.json").read_text(encoding="utf-8"))
    expected = {
        "manifest_sha256": _sha256_file(data_root / "manifest.jsonl"),
        "splits_sha256": _sha256_file(data_root / "splits.json"),
        "protocol": "V13_roi_metadata_stack", "seed": SEED, "epochs": epochs,
    }
    if any(signature.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved hip training protocol/data differ; choose a new checkpoint directory or resume=False")
    snapshot = Path(os.environ.get("HIP_MODEL_PATH", source_dir / "pretrained/resnet18"))
    provenance = _model_source_provenance(snapshot, None)
    if signature.get("model_source_provenance", {}).get("file_sha256") != provenance["file_sha256"]:
        raise ValueError("Saved hip training used different pretrained weights; use a new checkpoint directory")
    manifest_path = source_dir / "checkpoint_manifest.json"
    if manifest_path.is_file():
        checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = checkpoint_manifest.get("files", {})
        if set(files) != {path.name for path in _required_fold_checkpoints(source_dir, range(5))}:
            raise ValueError("Invalid hip training checkpoint manifest")
        for name, expected_hash in files.items():
            if _sha256_file(source_dir / name) != expected_hash:
                raise ValueError(f"Hip source checkpoint checksum mismatch: {name}")
    predictions = load_saved_predictions(data_root, source_dir)
    metadata = json.loads((source_dir / "metrics_and_run_metadata.json").read_text(encoding="utf-8"))
    return {"metrics": metadata["metrics"], "predictions": predictions,
            "metrics_path": source_dir / "metrics_and_run_metadata.json",
            "predictions_path": source_dir / "oof_predictions.csv", "output_dir": source_dir}


def train(data=None, *, checkpoints_dir=None, device="auto", epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    """Train V13 quality models and one side classifier, or resume verified fits.

    ``data`` is the prepared dataset directory containing manifest.jsonl,
    splits.json and images/. Every generated training artifact is saved under
    ``checkpoints_dir/source``. Successful training can export and validate an
    ONNX runtime under ``checkpoints_dir/runtime``. Logging is enabled only by
    ``verbose=True`` or ``ci=True``; no CI environment variable enables it.
    Resume is at complete outer-fold boundaries, never at an unfinished epoch.
    ``epochs`` changes the upper bound for every one of the 25 neural fits.
    ``resume=False`` removes only this trainer's known source artifacts and
    starts all folds again, leaving an existing runtime available until export
    successfully replaces it.
    """
    from ..common import Progress, resolve_checkpoint_paths, write_json

    if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0):
        raise ValueError("epochs must be a positive integer or None")
    data_root = (Path(data).expanduser().resolve() if data is not None
                 else Path(__file__).resolve().parents[1] / "data/dxa_v1")
    base, source_dir, runtime_dir = resolve_checkpoint_paths("hip", checkpoints_dir)
    progress = Progress("hip", enabled=verbose or ci)
    override = os.environ.get("HIP_MODEL_PATH")
    if override:
        snapshot = Path(override)
        if not snapshot.is_dir():
            raise FileNotFoundError(f"HIP_MODEL_PATH must name a local ResNet18 snapshot: {snapshot}")
    else:
        from .bootstrap import ensure_pretrained
        ensure_pretrained(source_dir / "pretrained/resnet18", verbose=verbose, ci=ci)
    for path in (data_root / "manifest.jsonl", data_root / "splits.json", data_root / "images"):
        if not path.exists():
            raise FileNotFoundError(f"Missing hip training input: {path}")
    if device == "auto":
        device = None

    global EPOCHS
    with _TRAIN_LOCK:
        previous_epochs = EPOCHS
        EPOCHS = previous_epochs if epochs is None else epochs
        token = _RUN_PROGRESS.set(progress)
        source_dir.mkdir(parents=True, exist_ok=True)
        try:
            progress.update("training", f"Checking dataset and checkpoints in {source_dir}")
            saved = _audit_complete_run(data_root, source_dir, EPOCHS) if resume else None
            resumed = saved is not None
            if not resume:
                for path in _source_artifact_paths(source_dir):
                    path.unlink(missing_ok=True)
            if saved is None:
                result = train_evaluate(data_root, source_dir, device=device)
            else:
                result = saved
                for fold in range(5):
                    progress.update("training", f"Fold {fold}: complete, checkpoints and OOF audited", fold + 1, 5)
            files = {path.name: _sha256_file(path)
                     for path in _required_fold_checkpoints(source_dir, range(5))}
            write_json(source_dir / "checkpoint_manifest.json", {
                "region": "hip", "protocol": "V13_roi_metadata_stack", "files": files,
                "manifest_sha256": _sha256_file(data_root / "manifest.jsonl"),
                "splits_sha256": _sha256_file(data_root / "splits.json"),
            })
            from .laterality.training import train as train_laterality
            progress.update("training", "Training/verifying automatic left-right classifier")
            side_report = train_laterality(data_root, source_dir=source_dir / "laterality",
                                           verbose=verbose, ci=ci, resume=resume, convert=False)
            export_result = None
            if convert:
                from .convert_all import convert_all
                export_result = convert_all(base, data=data_root, device=device or "auto",
                                            verbose=verbose, ci=ci)
            report = {
                "region": "hip", "checkpoints_dir": str(base), "source_dir": str(source_dir),
                "runtime_dir": str(runtime_dir), "resumed": resumed, "epochs": EPOCHS,
                "metrics": result["metrics"], "metrics_path": str(result["metrics_path"]),
                "predictions_path": str(result["predictions_path"]),
                "models": {"outer_folds": 5, "positioning_members": 5, "roi_neural_members": 20,
                           "laterality_classifier": 1, "laterality_frozen_resnet18": 1},
                "laterality": side_report,
                "converted": bool(convert), "conversion": export_result,
            }
            write_json(base / "training_report.json", report)
            progress.update("training", f"Complete; checkpoints saved in {base}")
            return report
        except BaseException as error:
            progress.update("training", f"Failed: {type(error).__name__}: {error}")
            raise
        finally:
            EPOCHS = previous_epochs
            _RUN_PROGRESS.reset(token)
