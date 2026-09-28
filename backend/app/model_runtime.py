"""Изолированный вызов исходного DXA-QC без изменения модельного ядра."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any

from app.schemas import AnatomicalRegion

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"
DEFAULT_MODELS_ROOT = Path(__file__).resolve().parents[1] / "models" / "reference"


class ReferenceRuntime:
    """Владеет единственным экземпляром эталонного ONNX-пайплайна.

    Args:
        models_root: Каталог локальных весов и манифеста.
        pipeline: Подставленный для тестов экземпляр с тем же контрактом.
        verify: Проверять установленный набор весов при загрузке.
    """

    def __init__(
        self,
        models_root: Path = DEFAULT_MODELS_ROOT,
        *,
        pipeline: Any | None = None,
        verify: bool = True,
    ) -> None:
        self.models_root = Path(models_root)
        self._pipeline = pipeline
        self._verify = verify
        self._lock = RLock()

    def _load(self) -> Any:
        if self._pipeline is None:
            if self._verify:
                from app.reference_assets import verify_models

                try:
                    verify_models(self.models_root)
                except ValueError as error:
                    raise ValueError(
                        "Локальный комплект моделей недоступен или повреждён"
                    ) from error
            import onnxruntime as ort

            if "CPUExecutionProvider" not in ort.get_available_providers():
                raise ValueError("CPUExecutionProvider недоступен")
            if str(REFERENCE_ROOT) not in sys.path:
                sys.path.insert(0, str(REFERENCE_ROOT))
            from combined_qc.router import QCPipeline

            self._pipeline = QCPipeline(
                checkpoints_dir=self.models_root / "router" / "checkpoints",
                hip_checkpoints_dir=self.models_root / "hip" / "checkpoints",
                spine_checkpoints_dir=self.models_root / "spine" / "checkpoints",
                device="cpu",
            )
            self._pipeline.load_checkpoints()
        return self._pipeline

    def ready(self) -> bool:
        """Проверяет установку моделей и холодную загрузку.

        Returns:
            Истина при доступности всех трёх модулей инференса.
        """
        with self._lock:
            try:
                self._load()
                return True
            except (OSError, RuntimeError, ValueError, ImportError):
                return False

    def infer(self, content: bytes, suffix: str, region: AnatomicalRegion) -> dict[str, Any]:
        """Запускает неизменённый runtime над временным оригинальным файлом.

        Args:
            content: Исходные байты снимка без изменения пикселей.
            suffix: Поддерживаемое расширение исходного формата.
            region: Автоматическая маршрутизация или ручной выбор профиля.

        Returns:
            Только разрешённые для API модельные поля без исходных путей.

        Raises:
            ValueError: Если формат или локальная поставка моделей недоступны.
        """
        if suffix.lower() not in {".dcm", ".dicom", ".png"}:
            raise ValueError("Неподдерживаемый формат для эталонной модели")
        with self._lock:
            return self._infer_locked(content, suffix, region)

    def _infer_locked(
        self, content: bytes, suffix: str, region: AnatomicalRegion
    ) -> dict[str, Any]:
        pipeline = self._load()
        with TemporaryDirectory(prefix="dexq-") as directory:
            path = Path(directory) / ("input" + suffix.lower())
            path.write_bytes(content)
            if region == AnatomicalRegion.AUTO:
                result = pipeline.infer(path)
                source = "model"
            else:
                name = {
                    AnatomicalRegion.LUMBAR_SPINE: "spine",
                    AnatomicalRegion.PROXIMAL_FEMUR: "hip",
                }.get(region)
                if name is None:
                    raise ValueError("Указана неподдерживаемая анатомическая область")
                if str(REFERENCE_ROOT) not in sys.path:
                    sys.path.insert(0, str(REFERENCE_ROOT))
                from combined_qc.hip import pipeline as hip_pipeline
                from combined_qc.spine import pipeline as spine_pipeline

                profile = spine_pipeline if name == "spine" else hip_pipeline
                handle = pipeline.models[name]
                result = profile.infer(path, device="cpu", model=handle)
                source = "operator"
        fields = (
            "region",
            "status",
            "labels",
            "scores",
            "any_violation",
            "needs_review",
            "laterality",
            "geometry",
            "image",
            "annotated_image",
            "criteria",
            "tasks",
            "routing",
        )
        allowed = {key: result[key] for key in fields if key in result}
        allowed["region_source"] = source
        return allowed
