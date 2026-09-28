# DEXQ

Локальный технический MVP для контроля качества DXA-снимков. Финальное эталонное решение работает за FastAPI и React без изменения модельного ядра; результаты — исследовательские, **не** клиническая валидация и не диагноз. Не загружайте DICOM и производные изображения во внешние сервисы.

## Запуск без Docker (проверен локально)

Нужны Python 3.12, Node.js 22 и **отдельный** финальный архив `dxa_qc_with_models.zip`, SHA-256 `2cd75f3777a4a7735d8e36cca371d4637b54053fb9283863c64c7214535a8867`. Веса **не входят в Git**; их 34 runtime-файла устанавливаются локально перед первым запуском.

Из каталога `backend/`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python ../scripts/prepare_reference.py /local/path/dxa_qc_with_models.zip
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Для Windows вместо `.venv/bin/python` используйте `.venv/Scripts/python`. Frontend в отдельном терминале:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

Веб-интерфейс: <http://localhost:3000>; документация API: <http://localhost:8000/docs>. Пока веса не установлены/не загрузились, `/api/v1/health` отвечает 503, не подменяя модель эвристикой. Для контейнеров, проверки манифеста и ограничений офлайн-поставки см. [развёртывание](docs/deployment.md); скрипт `scripts/run.sh` требует локальный Docker Engine, рабочий контейнерный запуск в текущем окружении пока **не подтверждён**.

## Демо и данные

Локально разложить 255 обезличенных примеров по экспертным категориям можно командой из `backend/`:

```bash
.venv/bin/python ../scripts/prepare_samples.py /local/path/dxa_qc_with_models.zip --output ../local-test-images
```

Папка игнорируется Git: `normal/`, `spine_positioning/`, `spine_axis/`, `spine_artifacts/`, `hip_positioning_rotation/`, `hip_roi/`, `multiple/`, `unlabeled/`. Один снимок может быть в нескольких категориях; это development-набор, не независимая проверка. [Руководство по экрану, ручной оси и CSV](docs/user-guide.md).

API принимает одиночный DICOM или демонстрационный grayscale PNG, пакет до 200 файлов и архив ZIP до 200 изображений для JSON или CSV; веб-интерфейс тоже принимает один ZIP. Экспорт содержит по одной строке на входное изображение, включая `Failure`, и восемь столбцов задания. Для локального CLI:

```bash
cd backend
.venv/bin/python -m app.cli /local/path/studies.zip /local/path/results.csv
```

Ограничения: проекция не определяется, бедро выдаёт объединённый флаг позиционирования/ротации, bbox бедра не анатомическая маска. Автоматический результат модели и ручная правка **двух найденных** точек позвоночника показаны отдельно; сервер исправления не хранит. Подробности: [спецификация](SPEC.md), [модель](docs/model.md), [метрики и доверительные интервалы](docs/metrics.md).

## Проверка изменений

```bash
cd backend
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
cd ../frontend
npm run build
node --experimental-strip-types --test src/geometry.test.ts
```

Для parity на исходных снимках укажите локальный `DEXQ_REFERENCE_ZIP=/local/path/dxa_qc_with_models.zip` перед pytest. Сохранённые development-метрики: F1 итогового OR **0.822** на 249 размеченных изображениях, study-bootstrap 95% CI **[0.747; 0.883]**; это не оценка генерализации. Локальный benchmark на 102 исследованиях/255 снимках дал 255/255 `Success`, максимум 11.4807 с/исследование на описанной Windows-конфигурации (цель `≤180 с` для этого development-набора выполнена). Офлайн Docker и перенос результата на другое железо не проверены.
