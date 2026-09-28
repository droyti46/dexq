"""HTTP API DEXQ версии 1."""

from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.analyzer import Analyzer
from app.batch import MAX_ARCHIVE_FILES, analyze_items, iter_archive, render_csv
from app.schemas import AnalysisResult, AnatomicalRegion, BatchItem, BatchResult, CheckInfo

router = APIRouter(prefix="/api/v1")
MAX_BATCH_FILES = MAX_ARCHIVE_FILES


def _analyzer(request: Request) -> Analyzer:
    return request.app.state.analyzer


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Файл превышает ограничение {max_bytes // (1024 * 1024)} МБ")
    return content


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Проверяет наличие всех локальных моделей перед началом обработки."""
    if not _analyzer(request).runtime.ready():
        raise HTTPException(status_code=503, detail="Локальный комплект моделей недоступен")
    return {"status": "ready", "service": "DEXQ API", "version": "0.1.0"}


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
        result = _analyzer(request).analyze(
            content, file.filename or "study.dcm", anatomical_region
        )
        if result.processing_status == "Failure":
            raise HTTPException(status_code=422, detail=result.error or "Анализ не завершён")
        return result
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
        raise HTTPException(
            status_code=422, detail=f"За один запрос принимается от 1 до {MAX_BATCH_FILES} файлов"
        )
    items: list[BatchItem] = []
    for file in files:
        filename = file.filename or "study.dcm"
        try:
            content = await _read_upload(file, request.app.state.max_upload_bytes)
        except ValueError as error:
            items.append(
                BatchItem(
                    filename=PurePath(filename.replace("\\", "/")).name,
                    input_path=filename,
                    error=str(error),
                )
            )
            continue
        items.extend(
            analyze_items(_analyzer(request), [(filename, content)], anatomical_region).items
        )
    successful = sum(
        item.result is not None and item.result.processing_status == "Success" for item in items
    )
    return BatchResult(items=items, successful=successful, failed=len(items) - successful)


@router.post("/analyses/batch.csv")
async def analyze_batch_csv(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    anatomical_region: Annotated[AnatomicalRegion, Form()] = AnatomicalRegion.AUTO,
) -> Response:
    """Возвращает пакетный отчёт в формате задания."""
    batch = await analyze_batch(request, files, anatomical_region)
    return Response(
        content=render_csv(batch),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="dexq-results.csv"'},
    )


@router.post("/analyses/archive", response_model=BatchResult)
async def analyze_archive(
    request: Request,
    archive: Annotated[UploadFile, File()],
    anatomical_region: Annotated[AnatomicalRegion, Form()] = AnatomicalRegion.AUTO,
) -> BatchResult:
    """Обрабатывает ZIP для браузера, сохраняя результаты и ошибки каждого файла."""
    try:
        content = await _read_upload(archive, request.app.state.max_archive_bytes)
        return analyze_items(
            _analyzer(request),
            iter_archive(content, request.app.state.max_upload_bytes),
            anatomical_region,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/analyses/archive.csv")
async def analyze_archive_csv(
    request: Request,
    archive: Annotated[UploadFile, File()],
    anatomical_region: Annotated[AnatomicalRegion, Form()] = AnatomicalRegion.AUTO,
) -> Response:
    """Обрабатывает ZIP тестового набора и возвращает единый CSV."""
    try:
        content = await _read_upload(archive, request.app.state.max_archive_bytes)
        batch = analyze_items(
            _analyzer(request),
            iter_archive(content, request.app.state.max_upload_bytes),
            anatomical_region,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Response(
        content=render_csv(batch),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="dexq-archive-results.csv"'},
    )
