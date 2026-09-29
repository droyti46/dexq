#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)
MODELS=${DEXQ_MODELS_DIR:-"$ROOT/backend/models/reference"}
PYTHON=${PYTHON:-python3}

if [ ! -d "$MODELS" ]; then
    printf 'Не найден локальный каталог моделей: %s\n' "$MODELS" >&2
    exit 1
fi

# Модели остаются на хосте; проверка хешей выполняется до запуска контейнеров.
DEXQ_VERIFY_ROOT="$ROOT/backend" "$PYTHON" - "$MODELS" <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["DEXQ_VERIFY_ROOT"])
from app.reference_assets import verify_models

try:
    verify_models(Path(sys.argv[1]))
except (OSError, ValueError) as error:
    raise SystemExit(f"Проверка моделей не пройдена: {error}") from error
PY

DEXQ_MODELS_DIR=$(CDPATH= cd -- "$MODELS" && pwd -P)
export DEXQ_MODELS_DIR
cd "$ROOT"
exec docker compose up --build
