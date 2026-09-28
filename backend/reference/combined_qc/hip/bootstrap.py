"""Fetch the exact public ImageNet backbone used by the hip V13 trainer."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

from ..common import Progress, publish_directory, write_json

MODEL_ID = "microsoft/resnet-18"
MODEL_REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
FILES = {
    "config.json": "11a01bcc873444bb433140e5f30f3687356641ed923266d738eef1c3438f6d22",
    "preprocessor_config.json": "fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741",
    "model.safetensors": "1cf00ee468998c23d084361b02cffadabacb5074105564c596b8079f77ecb126",
}
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "checkpoints/source/pretrained/resnet18"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_pretrained(snapshot_dir=None, *, verbose=False, ci=False):
    """Validate or download a pinned, checksum-checked local ResNet18 snapshot.

    Downloads use public Hugging Face resolve URLs and require no account or
    application token. A complete snapshot is published only after every file
    passes its SHA-256 check. Existing corrupt files are rejected explicitly.
    """
    destination = (Path(snapshot_dir).expanduser().resolve() if snapshot_dir is not None
                   else DEFAULT_SNAPSHOT)
    progress = Progress("hip", enabled=verbose or ci)
    missing = []
    for name, expected_hash in FILES.items():
        path = destination / name
        if not path.is_file():
            missing.append(name)
        elif _sha256(path) != expected_hash:
            raise ValueError(f"Pinned hip pretrained checksum mismatch: {path}")
    if not missing:
        progress.update("pretrained", f"Validated local {MODEL_ID}@{MODEL_REVISION}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".resnet18-download-", dir=destination.parent))
    try:
        for index, (name, expected_hash) in enumerate(FILES.items(), 1):
            target = staging / name
            if name not in missing:
                shutil.copy2(destination / name, target)
                continue
            # A custom checkpoint folder keeps its own complete initial snapshot.
            # Reuse the verified public cache without downloading it again.
            cached = DEFAULT_SNAPSHOT / name
            if cached.resolve() != (destination / name).resolve() and cached.is_file():
                if _sha256(cached) != expected_hash:
                    raise ValueError(f"Pinned hip pretrained cache mismatch: {cached}")
                shutil.copy2(cached, target)
                progress.update("pretrained", f"Copied verified {name}", index, len(FILES))
                continue
            progress.update("pretrained", f"Downloading {name}", index, len(FILES))
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "DXA-QC/1.0"})
            digest, received, last_update = hashlib.sha256(), 0, time.monotonic()
            with urllib.request.urlopen(request, timeout=45) as response, target.open("wb") as handle:
                expected_size = response.headers.get("Content-Length")
                total = int(expected_size) if expected_size and expected_size.isdigit() else None
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(chunk)
                    if received > 512 * 1024 * 1024:
                        raise ValueError(f"Unexpectedly large pretrained download: {name}")
                    digest.update(chunk)
                    handle.write(chunk)
                    if time.monotonic() - last_update >= 3:
                        progress.update("pretrained", name, received, total)
                        last_update = time.monotonic()
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Downloaded pretrained checksum mismatch: {name}")
            progress.update("pretrained", f"{name}: SHA-256 verified ({received} bytes)", index, len(FILES))
        write_json(staging / "download_provenance.json", {
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": FILES,
        })
        publish_directory(staging, destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
