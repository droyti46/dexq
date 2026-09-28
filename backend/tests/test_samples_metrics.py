"""Каталог локальных примеров и интервальные метрики без построчных публикаций."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from metrics import bootstrap_ci, evaluate_rows  # noqa: E402
from prepare_samples import prepare_samples  # noqa: E402


def _make_private_zip(path: Path) -> None:
    payload = b"DICOM demo bytes"
    source = "case/patient_SECRET.dcm"
    records = [
        {
            "image_id": "img_001",
            "source": source,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "expected_region": "spine",
            "study_id": "study_1",
            "truth": {
                "labels": {"spine_axis": True, "spine_positioning": True, "spine_artifacts": False},
                "any_violation": True,
            },
            "labels": {"spine_axis": True, "spine_positioning": True, "spine_artifacts": False},
            "scores": {"spine_axis": 7.0, "spine_positioning": 0.7, "spine_artifacts": 0.1},
            "any_violation": True,
        }
    ]
    with ZipFile(path, "w") as archive:
        archive.writestr("dxa_qc_code/source_data/" + source, payload)
        archive.writestr(
            "dxa_qc_code/combined_qc/outputs/deployment_metrics.json",
            json.dumps({"per_image": records}),
        )


def test_multilabel_sample_has_each_category_without_original_name(tmp_path: Path) -> None:
    archive = tmp_path / "samples.zip"
    _make_private_zip(archive)
    counts = prepare_samples(archive, tmp_path / "output")
    assert counts["spine_axis"] == counts["spine_positioning"] == counts["multiple"] == 1
    assert (tmp_path / "output/spine_axis/case_0001.dcm").read_bytes() == b"DICOM demo bytes"
    assert "patient_SECRET" not in str(list((tmp_path / "output").rglob("*")))
    assert prepare_samples(archive, tmp_path / "output") == counts


def test_study_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        {"study_id": "one", "truth": True, "prediction": True},
        {"study_id": "one", "truth": True, "prediction": True},
        {"study_id": "two", "truth": False, "prediction": False},
    ]
    ci = bootstrap_ci(rows, seed=7, repeats=40)
    assert ci == bootstrap_ci(rows, seed=7, repeats=40)
    assert ci["f1"]["valid"] < 40
    assert ci["f1"]["invalid"] > 0


def test_auc_uses_probability_ranking_and_ties() -> None:
    rows = [
        {"study_id": "one", "truth": True, "prediction": True, "score": .9},
        {"study_id": "two", "truth": False, "prediction": False, "score": .1},
        {"study_id": "three", "truth": False, "prediction": True, "score": .9},
    ]
    assert evaluate_rows(rows)["roc_auc"] == .75


def test_aggregate_metrics_excludes_undefined_auc() -> None:
    rows = [{"study_id": "only", "truth": True, "prediction": True, "score": 0.9}]
    metrics = evaluate_rows(rows)
    assert metrics["f1"] == 1
    assert metrics["roc_auc"] is None
    assert metrics["positives"] == 1
