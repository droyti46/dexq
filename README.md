# DEXQ

Локальный медицинский инструмент для автоматической проверки качества DXA-исследований. Проект состоит из React-интерфейса и FastAPI-сервиса с независимыми модулями контроля укладки, охвата и артефактов.

> Текущая версия — технический MVP для хакатона. Эвристические проверки не прошли клиническую валидацию и не предназначены для постановки диагноза.

Полная продуктовая и техническая спецификация находится в [SPEC.md](./SPEC.md).

## Быстрый запуск без Docker

Требуются Python 3.12 и Node.js 22.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Документация API: <http://localhost:8000/docs>.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Интерфейс: <http://localhost:3000>.

## Запуск в контейнерах

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

Или напрямую:

```bash
docker compose up --build
```

Frontend откроется на <http://localhost:3000>, API — на <http://localhost:8000>.

## API

- `GET /api/v1/health` — проверка готовности.
- `GET /api/v1/checks` — реестр проверок.
- `POST /api/v1/analyses` — один DICOM.
- `POST /api/v1/analyses/batch` — пакет до трёх файлов.
- `POST /api/v1/analyses/batch.csv` — пакетный CSV по формату задания.

Для демонстрации одиночный endpoint также принимает PNG/JPEG. Анатомическую область можно передать полем `anatomical_region`: `auto`, `lumbar_spine` или `proximal_femur`.

## Структура

```text
backend/             FastAPI, чтение DICOM, проверки и тесты
frontend/            React/Vite, основные экраны
data/                исходные архивы организатора (не копируются в контейнер)
scripts/run.sh       запуск контейнерного стенда
SPEC.md              единая спецификация
CLAUDE.md            правила для AI-разработчиков
spine_pipeline.ipynb исследовательский pipeline позвоночника
```

## Проверка

```powershell
cd backend
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .

cd ..\frontend
npm run build
```

## Известные ограничения

- Верхняя ветка охвата Th12, SpineNet и классификатор артефактов из notebook ещё не подключены к runtime.
- Ось позвоночника в MVP локализуется временной детерминированной эвристикой; правило угла и порог 5° соответствуют notebook.
- Модули бедра уже представлены в общем контракте, но до подключения моделей возвращают `not_evaluated`.
- Автоопределение области основано на DICOM-описаниях и может потребовать ручного выбора.

