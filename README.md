# DEXQ

Локальный контроль качества DXA-снимков: поясничный отдел позвоночника и проксимальный отдел бедра. Сервис проверяет укладку, наклон оси, артефакты и охват зоны интереса. Работает на CPU, снимки не покидают машину.

**📘 Документация: <https://droyti46.github.io/dexq/>** — запуск, сайт со скриншотами, API и устройство кода.

> Исследовательский инструмент, а не медицинское изделие. Метрики получены на development-наборе и не являются клинической валидацией. Не загружайте DICOM и производные изображения во внешние сервисы.

## Запуск

DEXQ передаётся одним архивом: код, веса моделей и конфигурация Docker. Скачивать ничего отдельно не нужно.

### В Docker

```bash
unzip dexq.zip && cd dexq
./scripts/run.sh
```

`run.sh` проверяет SHA-256 всех весов в `backend/models/reference/` и запускает `docker compose up --build`. На Windows (PowerShell):

```powershell
$env:DEXQ_MODELS_DIR = "$PWD\backend\models\reference"
eference"
docker compose up --build
```

### Без Docker

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Во втором терминале из корня репозитория:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

На Windows вместо `.venv/bin/python` используйте `.venv/Scripts/python`.

### Адреса

| Что | Адрес |
| --- | --- |
| Сайт | <http://localhost:3000> |
| Swagger UI | <http://localhost:8000/docs> |
| Готовность | <http://localhost:8000/api/v1/health> |

В Git-репозитории весов нет: для чистой копии из GitHub их устанавливает `scripts/prepare_reference.py` из эталонного архива `dxa_qc_with_models (2).zip` (SHA-256 `b92ed6e89b1bb54358b1e06681842dbecac006880cf9062ea68980673ff12f0a`).

### Пакетная обработка без сайта

```bash
cd backend
.venv/bin/python -m app.cli /путь/к/studies.zip /путь/к/results.csv
```

## Проверка изменений

```bash
cd backend && .venv/bin/python -m pytest && .venv/bin/python -m ruff check .
cd frontend && npm run build && node --experimental-strip-types --test src/*.test.ts
```

## Документация локально

```bash
pip install -r docs/requirements.txt
mkdocs serve -a 127.0.0.1:8001
```

Исходники страниц — в [`docs/`](docs/), спецификация — в [`SPEC.md`](SPEC.md).

Команда «Люди в черном».
