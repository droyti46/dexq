"""Точка входа FastAPI-приложения DEXQ."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analyzer import Analyzer
from app.api import router


def create_app() -> FastAPI:
    """Создаёт приложение и явные зависимости верхнего уровня.

    Returns:
        Настроенное FastAPI-приложение.
    """
    app = FastAPI(
        title="DEXQ API",
        description="Локальный контроль качества DXA-исследований",
        version="0.1.0",
    )
    app.state.analyzer = Analyzer()
    app.state.max_upload_bytes = int(os.getenv("DEXQ_MAX_UPLOAD_MB", "50")) * 1024 * 1024
    app.state.max_archive_bytes = int(os.getenv("DEXQ_MAX_ARCHIVE_MB", "1024")) * 1024 * 1024
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"title": "DEXQ API", "docs": "/docs", "status": "ok"}

    return app


app = create_app()
