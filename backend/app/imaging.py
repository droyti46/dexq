"""Чтение медицинских изображений без раскрытия персональных DICOM-тегов."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_voi_lut

from app.schemas import AnatomicalRegion


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Безопасное представление загруженного изображения."""

    pixels: np.ndarray
    study_uid: str | None
    image_uid: str | None
    region: AnatomicalRegion
    region_source: str
    preview_data_url: str


def load_medical_image(content: bytes, filename: str) -> LoadedImage:
    """Декодирует DICOM или демонстрационное растровое изображение.

    Args:
        content: Полное содержимое загруженного файла.
        filename: Исходное имя файла, используемое только для определения формата.

    Returns:
        Нормализованное восьмибитное изображение и безопасные метаданные.

    Raises:
        ValueError: Если файл пуст, повреждён или не содержит поддерживаемых пикселей.
    """
    if not content:
        raise ValueError("Файл пуст")

    suffix = Path(filename).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _load_raster(content)
    return _load_dicom(content, filename)


def _load_dicom(content: bytes, filename: str) -> LoadedImage:
    try:
        dataset = pydicom.dcmread(BytesIO(content), force=False)
        if "PixelData" not in dataset:
            raise ValueError("DICOM не содержит PixelData")
        raw = np.asarray(apply_voi_lut(dataset.pixel_array, dataset))
    except (InvalidDicomError, ValueError, TypeError, AttributeError, EOFError, OSError) as error:
        raise ValueError(f"Не удалось прочитать DICOM: {error}") from error

    pixels = _to_grayscale(raw)
    normalized = _normalize_uint8(pixels)
    if str(dataset.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        normalized = 255 - normalized
    region, source = _detect_region(dataset, filename)
    return LoadedImage(
        pixels=normalized,
        study_uid=_safe_uid(dataset.get("StudyInstanceUID")),
        image_uid=_safe_uid(dataset.get("SOPInstanceUID")),
        region=region,
        region_source=source,
        preview_data_url=_preview_data_url(normalized),
    )


def _load_raster(content: bytes) -> LoadedImage:
    try:
        with Image.open(BytesIO(content)) as image:
            pixels = np.asarray(image.convert("L")).copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"Не удалось прочитать изображение: {error}") from error
    if pixels.size == 0:
        raise ValueError("Изображение не содержит пикселей")
    return LoadedImage(
        pixels=pixels,
        study_uid=None,
        image_uid=None,
        region=AnatomicalRegion.UNKNOWN,
        region_source="not_available_for_raster",
        preview_data_url=_preview_data_url(pixels),
    )


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        rgb = array[..., :3].astype(np.float32)
        return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if array.ndim == 3:
        return array[0]
    if array.ndim == 4:
        return _to_grayscale(array[0])
    raise ValueError(f"Неподдерживаемая размерность пикселей: {array.shape}")


def _normalize_uint8(array: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(array, dtype=np.float32))
    if values.size == 0 or min(values.shape) < 8:
        raise ValueError("Изображение слишком мало для анализа")
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) * (255 / (high - low)), 0, 255).astype(np.uint8)


def _detect_region(dataset: Dataset, filename: str) -> tuple[AnatomicalRegion, str]:
    fields = (
        "BodyPartExamined",
        "StudyDescription",
        "SeriesDescription",
        "ProtocolName",
        "PerformedProcedureStepDescription",
    )
    description = " ".join(str(dataset.get(name, "")) for name in fields).lower()
    spine_tokens = ("spine", "lumbar", "l-spine", "пояснич", "позвоноч")
    hip_tokens = ("hip", "femur", "femoral", "бедр", "тазобед")
    if any(token in description for token in spine_tokens):
        return AnatomicalRegion.LUMBAR_SPINE, "dicom_metadata"
    if any(token in description for token in hip_tokens):
        return AnatomicalRegion.PROXIMAL_FEMUR, "dicom_metadata"
    normalized_name = Path(filename).stem.lower()
    if "поп" in normalized_name or "lumbar" in normalized_name or "spine" in normalized_name:
        return AnatomicalRegion.LUMBAR_SPINE, "filename"
    hip_name_tokens = ("ппоб", "лпоб", "hip", "femur")
    if any(token in normalized_name for token in hip_name_tokens):
        return AnatomicalRegion.PROXIMAL_FEMUR, "filename"
    return AnatomicalRegion.UNKNOWN, "dicom_metadata_inconclusive"


def _safe_uid(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and len(text) <= 128 else None


def _preview_data_url(pixels: np.ndarray) -> str:
    image = Image.fromarray(pixels, mode="L")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def content_id(content: bytes) -> str:
    """Возвращает стабильный неперсональный идентификатор содержимого.

    Args:
        content: Байты исходного файла.

    Returns:
        Короткий SHA-256 идентификатор.
    """
    return "an_" + hashlib.sha256(content).hexdigest()[:20]
