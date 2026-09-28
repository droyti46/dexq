# DEXQ

Локальный технический MVP для контроля качества DXA-снимков. Финальное эталонное решение работает за FastAPI и React без изменения модельного ядра; результаты — исследовательские, **не** клиническая валидация и не диагноз. Не загружайте DICOM и производные изображения во внешние сервисы.

## Запуск без Docker (проверен локально)

Нужны Python 3.12, Node.js 22 и **отдельный** оптимизированный архив `dxa_qc_with_models (2).zip`, SHA-256 `b92ed6e89b1bb54358b1e06681842dbecac006880cf9062ea68980673ff12f0a`. Веса **не входят в Git**; 22 runtime-файла (276 423 364 байта вместе с манифестами и конфигурацией) устанавливаются локально перед первым запуском.

Из каталога `backend/`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python ../scripts/prepare_reference.py "/local/path/dxa_qc_with_models (2).zip"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Для Windows вместо `.venv/bin/python` используйте `.venv/Scripts/python`. Frontend в отдельном терминале:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

Веб-интерфейс: <http://localhost:3000>; документация API: <http://localhost:8000/docs>. Пока веса не установлены/не загрузились, `/api/v1/health` отвечает 503; fallback на другую модель не предусмотрен. Для контейнеров, проверки манифеста и ограничений офлайн-поставки см. [развёртывание](docs/deployment.md); скрипт `scripts/run.sh` требует локальный Docker Engine, рабочий контейнерный запуск в текущем окружении пока **не подтверждён**.

## Демо и данные

Локально разложить 255 обезличенных примеров по экспертным категориям можно командой из `backend/`:

```bash
.venv/bin/python ../scripts/prepare_samples.py "/local/path/dxa_qc_with_models (2).zip" --output ../local-test-images
```

Папка игнорируется Git: `normal/`, `spine_positioning/`, `spine_axis/`, `spine_artifacts/`, `hip_positioning_rotation/`, `hip_roi/`, `multiple/`, `unlabeled/`. Один снимок может быть в нескольких категориях; это development-набор, не независимая проверка. [Руководство по экрану, ручной оси и CSV](docs/user-guide.md).

API принимает одиночный DICOM или демонстрационный grayscale PNG, пакет до 200 файлов и архив ZIP до 200 изображений для JSON или CSV; веб-интерфейс тоже принимает один ZIP. Экспорт содержит по одной строке на входное изображение, включая `Failure`, и восемь столбцов задания. Для локального CLI:

```bash
cd backend
.venv/bin/python -m app.cli /local/path/studies.zip /local/path/results.csv
```

Ограничения: проекция не определяется, бедро выдаёт объединённый флаг позиционирования/ротации, bbox бедра не анатомическая маска. Автоматический результат модели и ручная правка **двух найденных** точек позвоночника показаны отдельно; сервер исправления не хранит. Разбивку проверок и границы интерпретации см. в [спецификации](SPEC.md) и [описании модели](docs/model.md); численные оценки — в [метриках](docs/metrics.md).

## Проверка изменений

```bash
cd backend
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
cd ../frontend
npm run build
node --experimental-strip-types --test src/geometry.test.ts
```

Для parity на исходных снимках укажите локальный `DEXQ_REFERENCE_ZIP` с путём к архиву перед pytest. Данные из архива — development-набор, не независимая клиническая проверка; описание оценок см. в [метриках](docs/metrics.md). Новый локальный benchmark текущего комплекта на 102 исследованиях/255 уникальных DICOM: 255/255 `Success`, максимум 15.895 с/исследование (≤3 кадра), среднее 4.262 с на прогретом CPUExecutionProvider, холодная загрузка 2.6479 с. Это те же development-снимки, а не независимая проверка точности или аппаратных требований; прежний benchmark относился к предыдущим весам. Офлайн Docker-запуск не подтверждён.
