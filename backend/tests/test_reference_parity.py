"""Сравнение API-адаптера с неизменённым эталоном на локальных DICOM."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pytest

from app.model_runtime import ReferenceRuntime
from app.schemas import AnatomicalRegion

REFERENCE_ZIP = Path(os.environ["DEXQ_REFERENCE_ZIP"]) if os.getenv("DEXQ_REFERENCE_ZIP") else None
MODELS = Path(__file__).resolve().parents[1] / "models" / "reference"


@pytest.mark.model
def test_reference_adapter_matches_original_for_each_profile() -> None:
    if (
        REFERENCE_ZIP is None
        or not REFERENCE_ZIP.is_file()
        or not (MODELS / "installed-manifest.json").is_file()
    ):
        pytest.skip("Локальный эталонный архив/веса не установлены")
    runtime = ReferenceRuntime(MODELS)
    assert runtime.ready()
    pipeline = runtime._load()
    with ZipFile(REFERENCE_ZIP) as archive:
        rows = json.loads(archive.read("dxa_qc_code/combined_qc/outputs/deployment_metrics.json"))[
            "per_image"
        ]
        cases = (
            next(row for row in rows if row["expected_region"] == "spine"),
            next(row for row in rows if row["truth"]["laterality"] == "left_hip"),
            next(row for row in rows if row["truth"]["laterality"] == "right_hip"),
        )
        names_by_sha = {
            hashlib.sha256(archive.read(name)).hexdigest(): name
            for name in archive.namelist()
            if name.startswith("dxa_qc_code/source_data/") and name.lower().endswith(".dcm")
        }
        for case in cases:
            content = archive.read(names_by_sha[case["source_sha256"]])
            with TemporaryDirectory() as directory:
                path = Path(directory) / "input.dcm"
                path.write_bytes(content)
                original = pipeline.infer(path)
            adapted = runtime.infer(content, ".dcm", AnatomicalRegion.AUTO)
            assert adapted["region"] == original["region"]
            assert adapted.get("laterality") == original.get("laterality")
            assert adapted["labels"] == original["labels"]
            assert adapted["any_violation"] == original["any_violation"]
            for name, score in original["scores"].items():
                if score is not None:
                    np.testing.assert_allclose(adapted["scores"][name], score, atol=1e-5, rtol=1e-4)
