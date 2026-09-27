"""HTTP API DEXQ версии 1."""

import csv
from io import StringIO
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.analyzer import Analyzer
from app.schemas import AnalysisResult, AnatomicalRegion, BatchItem, BatchResult, CheckInfo

router = APIRouter(prefix="/api/v1")
MAX_BATCH_FILES = 3


def _analyzer(request: Request) -> Analyzer:
    return request.app.state.analyzer


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Файл превышает ограничение {max_bytes // (1024 * 1024)} МБ")
    return content


@router.get("/health")
async def health() -> dict[str, str]:
    """Возвращает готовность API."""
    return {"status": "ok", "service": "DEXQ API", "version": "0.1.0"}


@router.get("/checks", response_model=list[CheckInfo])
async def list_checks(request: Request) -> list[CheckInfo]:
    """Возвращает список модулей контроля качества."""
    return _analyzer(request).describe_checks()


@router.post("/analyses", response_model=AnalysisResult)
async def analyze_file(
    request: Request,
    file: Annotated[UploadFile, File()],
    anatomical_region: Annotated[AnatomicalRegion, Form()] = AnatomicalRegion.AUTO,
) -> AnalysisResult:
    """Обрабатывает один файл полностью в памяти."""
    try:
        content = await _read_upload(file, request.app.state.max_upload_bytes)
        return _analyzer(request).analyze(content, file.filename or "study.dcm", anatomical_region)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/analyses/batch", response_model=BatchResult)
async def analyze_batch(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    anatomical_region: Annotated[AnatomicalRegion, Form()] = AnatomicalRegion.AUTO,
) -> BatchResult:
    """Обрабатывает пакет, не прерываясь из-за ошибки отдельного файла."""
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(status_code=422, detail="За один запрос принимается от 1 до 3 файлов")
    items: list[BatchItem] = []
    for file in files:
        filename = file.filename or "study.dcm"
        try:
            content = await _read_upload(file, request.app.state.max_upload_bytes)
            result = _analyzer(request).analyze(content, filename, anatomical_region)
            items.append(BatchItem(filename=filename, result=result))
        except ValueError as error:
            items.append(BatchItem(filename=filename, error=str(error)))
    successful = sum(item.result is not None for item in items)
    return BatchResult(items=items, successful=successful, failed=len(items) - successful)


@router.post("/analyses/batch.csv")
async def analyze_batch_csv(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    anatomical_region: Annotated[AnatomicalRegion, Form()] = AnatomicalRegion.AUTO,
) -> Response:
    """Возвращает пакетный отчёт в формате задания."""
    batch = await analyze_batch(request, files, anatomical_region)
    output = StringIO(newline="")
    fieldnames = [
        "path_to_study",
        "study_uid",
        "image_uid",
        "anatomical_region",
        "quality_class",
        "violation_type",
        "processing_status",
        "time_of_processing",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in batch.items:
        result = item.result
        writer.writerow(
            {
                "path_to_study": item.filename,
                "study_uid": result.study_uid if result else "",
                "image_uid": result.image_uid if result else "",
                "anatomical_region": result.anatomical_region if result else "unknown",
                "quality_class": result.quality_class if result else "",
                "violation_type": ";".join(result.violation_types) if result else item.error,
                "processing_status": result.processing_status if result else "Failure",
                "time_of_processing": result.time_of_processing if result else 0,
            }
        )
    return Response(
        content="\ufeff" + output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="dexq-results.csv"'},
    )
