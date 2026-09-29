"""Свойства адаптера к эталонному локальному инференсу."""

from pathlib import Path

import pytest

from app.model_runtime import ReferenceRuntime
from app.schemas import AnatomicalRegion


class FakePipeline:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.models = {"spine": object(), "hip": object()}

    def load_checkpoints(self) -> None:
        pass

    def infer(self, source: Path) -> dict:
        self.paths.append(Path(source))
        return {
            "source": str(source),
            "region": "spine",
            "labels": {"spine_axis": False},
            "scores": {"spine_axis": 1.5},
            "any_violation": False,
            "routing": {"region": "spine", "confidence": 0.99},
            "geometry": {"image_width": 320, "image_height": 320},
            "image": {"encoding": "base64_png", "data": "YQ==", "mode": "L"},
            "raw": {"PatientName": "SECRET"},
            "metadata": {"study_id": "SECRET"},
        }


def test_missing_models_cannot_be_ready(tmp_path: Path) -> None:
    runtime = ReferenceRuntime(tmp_path / "missing")
    assert runtime.ready() is False
    with pytest.raises(ValueError, match="модел"):
        runtime.infer(b"pixels", ".dcm", AnatomicalRegion.AUTO)


def test_auto_uses_one_pipeline_and_removes_temp_input(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    runtime = ReferenceRuntime(tmp_path, pipeline=pipeline, verify=False)
    result = runtime.infer(b"original", ".dcm", AnatomicalRegion.AUTO)
    assert len(pipeline.paths) == 1
    assert pipeline.paths[0].name == "input.dcm"
    assert not pipeline.paths[0].exists()
    assert result["labels"] == {"spine_axis": False}
    assert result["routing"]["confidence"] == 0.99
    assert "PatientName" not in str(result)
    assert "SECRET" not in str(result)
    assert "source" not in result
    assert "raw" not in result
    assert "metadata" not in result


def test_invalid_suffix_rejected_without_writing(tmp_path: Path) -> None:
    runtime = ReferenceRuntime(tmp_path, pipeline=FakePipeline(), verify=False)
    with pytest.raises(ValueError, match="формат"):
        runtime.infer(b"pixels", ".exe", AnatomicalRegion.AUTO)
