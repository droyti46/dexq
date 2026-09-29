"""Restore the pinned public pretrained backbones without private weight bundles."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .engine.utils import sha256
from ..common import Progress

RESNET_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET_SOURCE_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET_STATE_SHA = "6125408e356323c4193a8ab1f56bb155aad05af3fc2b542e3c5a8587e9a17505"
SPINE_DRIVE_ID = "1X9gsP9_tvjPrfFlFWKkb9oP9jb-XmN7S"
SPINE_ZIP_SHA = "1b088004614c09ad6605f444c39b160144b07704b58bae13f21dd19665d7a1d2"
SPINE_SHA = "6a779e01b9a41601334e0a9541278fc557a95bd650c6c8de311204821509d19b"


def ensure_sources(source_dir, *, verbose=False, ci=False):
    source = Path(source_dir).resolve()
    source.mkdir(parents=True, exist_ok=True)
    progress = Progress("spine", verbose or ci)
    target = source / "resnet18.pt"
    meta_path = target.with_suffix(".json")
    ready = False
    if target.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        ready = meta.get("state_sha256") == RESNET_STATE_SHA and meta.get("file_sha256") == sha256(target)
        if not ready:
            raise ValueError("Existing ResNet18 checkpoint failed integrity validation")
    if not ready:
        import torch
        from torch import nn
        from torchvision.models import resnet18
        from .engine.features import state_digest
        progress.update("bootstrap", "Downloading pinned public ResNet18", 0, 2)
        with tempfile.TemporaryDirectory(dir=source, prefix="download-") as temp:
            raw = Path(temp) / "official.pth"
            with urlopen(Request(RESNET_URL, headers={"User-Agent": "dxa-qc/1.0"}), timeout=120) as response, raw.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha256(raw) != RESNET_SOURCE_SHA:
                raise ValueError("Public ResNet18 download checksum mismatch")
            full = resnet18(weights=None)
            full.load_state_dict(torch.load(raw, map_location="cpu", weights_only=True), strict=True)
            features = nn.Sequential(*list(full.children())[:-2])
            if state_digest(features) != RESNET_STATE_SHA:
                raise ValueError("Rebuilt ResNet18 numerical state differs")
            rebuilt = Path(temp) / "resnet18.pt"
            torch.save(features.state_dict(), rebuilt)
            meta = {"kind": "resnet", "name": "resnet18", "source": {
                "url": RESNET_URL, "source_sha256": RESNET_SOURCE_SHA,
                "weights": "ResNet18_Weights.IMAGENET1K_V1"},
                "state_sha256": RESNET_STATE_SHA, "file_sha256": sha256(rebuilt)}
            rebuilt.replace(target)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    progress.update("bootstrap", "ResNet18 checkpoint verified", 1, 2)
    target = source / "vertebra_landmark.pth"
    if target.is_file():
        if sha256(target) != SPINE_SHA:
            raise ValueError("Existing SpineNet checkpoint failed integrity validation")
    else:
        import gdown
        progress.update("bootstrap", "Downloading pinned public SpineNet", 1, 2)
        with tempfile.TemporaryDirectory(dir=source, prefix="download-") as temp:
            zipped = Path(temp) / "weights.zip"
            gdown.download(id=SPINE_DRIVE_ID, output=str(zipped), quiet=True, use_cookies=False)
            if not zipped.is_file() or sha256(zipped) != SPINE_ZIP_SHA:
                raise ValueError("Public SpineNet archive checksum mismatch")
            with ZipFile(zipped) as archive:
                if archive.namelist() != ["model_last.pth"]:
                    raise ValueError("Unexpected public SpineNet archive layout")
                rebuilt = Path(temp) / "vertebra_landmark.pth"
                with archive.open("model_last.pth") as src, rebuilt.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if sha256(rebuilt) != SPINE_SHA:
                raise ValueError("Public SpineNet checkpoint checksum mismatch")
            rebuilt.replace(target)
    progress.update("bootstrap", "SpineNet checkpoint verified", 2, 2)
    return source
