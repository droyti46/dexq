# Локальное развёртывание DEXQ

## Состояние проверки

Офлайн-установка весов и предзапускная проверка SHA-256 выполнены локально. Конфигурация Compose проверена командой `docker compose config`; **запуск/healthcheck контейнеров ещё не подтверждён**, поскольку Docker Desktop на проверяющей машине не предоставил рабочий Linux Engine (ошибка API 500). До успешного контейнерного запуска это не готовый проверенный офлайн-комплект для судьи. Образам для первоначальной сборки нужны локальные слои и кэши `pip`/`npm` либо доступ в сеть; флаг `--build` сам по себе не обеспечивает офлайн-сборку. Не объявлять поставку полностью автономной без сохранённых готовых образов и проверки на отдельном компьютере без сети.

## Требования и источник

Python 3.12 для подготовки весов, Docker Engine с Compose для контейнеров и локальный оптимизированный архив `dxa_qc_with_models (2).zip` с SHA-256 `b92ed6e89b1bb54358b1e06681842dbecac006880cf9062ea68980673ff12f0a`. Другие hip-веса не используются. Из архива извлекаются 22 runtime-файла (276 423 364 байта, ≈264 MiB, включая JSON-манифесты) в `backend/models/reference/`; исходный архив и веса не входят в Git и Docker build context. Директория монтируется только для чтения по `/models/reference`. Образ backend содержит 74 Python-файла `combined_qc`, совпадающих с архивом побайтово, но не содержит весов/изображений.

Базовые образы привязаны к multiarch digest:

| Этап | Имя и digest |
| --- | --- |
| Python | `python:3.12.6-slim-bookworm@sha256:ad48727987b259854d52241fac3bc633574364867b8e20aec305e6e7f4028b26` |
| Node build | `node:22.23.0-alpine3.23@sha256:35e2f96595091599e7c1fb0b61049e17d8478997b2aa13db51ad7995299fe55a` |
| Nginx | `nginx:1.27.5-alpine3.21@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10` |

Backend-зависимости зафиксированы в `backend/requirements.lock`, сборочная версия setuptools — в `backend/pyproject.toml`; frontend — в `frontend/package-lock.json`. Основной ONNX provider — `CPUExecutionProvider`, GPU не требуется и не проверен. Привязка digest не означает, что образы доступны без сети на чистой машине.

## Запуск на Linux/Unix

Из корня распакованного исходного кода, имея локальный архив с весами:

```bash
PYTHONPATH=backend python3.12 scripts/prepare_reference.py "/local/path/dxa_qc_with_models (2).zip" --source backend/reference --models backend/models/reference
```

Запустите команду из `backend/` как `python3.12 ../scripts/prepare_reference.py "/local/path/dxa_qc_with_models (2).zip"`: CLI-обёртка использует пакет `app` из текущего Python-окружения. При запуске из корня проекта установите `PYTHONPATH=backend`. Установите backend-зависимости до извлечения, если среда ещё не подготовлена.

```bash
PYTHON=python3.12 ./scripts/run.sh
```

По умолчанию скрипт проверяет `backend/models/reference/installed-manifest.json`, SHA исходной поставки и хеши всех файлов, экспортирует абсолютный `DEXQ_MODELS_DIR` и только затем вызывает `docker compose up --build`. Для отдельной директории используйте `DEXQ_MODELS_DIR=/absolute/local/models/reference`. Compose монтирует её `:ro`. `GET http://localhost:8000/api/v1/health` возвращает готовность только после загрузки моделей; веб-интерфейс — `http://localhost:3000`. Установка/запуск не передают медицинские файлы наружу; самостоятельная сборка образы и Python/npm-зависимости может загружать из публичных реестров, если они не кэшированы. Не используйте `docker compose up` напрямую без экспорта переменной, если ожидаете проверку весов.

## Поставка судьям (целевая структура)

```text
submission/
  source/                       исходники и конфигурация без медицинских данных
  models/reference/             все 22 локальных runtime-файла + installed-manifest.json
  images/                       заранее собранные сохранённые Docker-образы для офлайн-хоста
  source/scripts/run.sh         локальная точка запуска и проверка SHA
```

При доступном Docker Engine сохраните готовые локальные образы (`docker save -o submission/dexq-images.tar dexq-backend:latest dexq-frontend:latest` после проверенной сборки), затем из каталога `backend/` выполните `python ../scripts/package_submission.py --output ../submission/dexq-offline.zip --models models/reference --images ../submission/dexq-images.tar`. Упаковщик включает только разрешённые отслеживаемые исходники, проверенные модели и сохранённые образы, исключает `data/` и изображения. После распаковки перенесите `models/reference/` в `source/backend/models/reference/`, импортируйте образы `docker load -i images/dexq-images.tar`, затем запустите `source/scripts/run.sh` из распакованного исходного дерева. Полный офлайн-архив с сохранёнными образами ещё **не сформирован и не испытан**. В Git/внешние сервисы не загружать `submission/`, DICOM, PNG, ONNX/NPZ, а также построчные метрики. Для передачи набора весов и/или архивов использовать только согласованный локальный канал.

## Ресурсы и время

Установленный runtime текущего архива содержит 22 файла (276 423 364 байта, ≈264 MiB); к месту на диске дополнительно нужны образы, временные входы и сама поставка. На локальном Windows 11 / Python 3.12.6 / CPUExecutionProvider с i7-1355U и 15.63 GiB RAM после удаления повторяющихся копий одного DICOM из экспертных папок и одного некорректного тестового файла получены 255 уникальных DICOM из 102 исследований (1–3 кадра): 255/255 `Success`, максимум **15.895 с/исследование**, среднее **4.262 с**, холодная загрузка **2.6479 с**. Измерение включает локальное чтение, декодирование, инференс, DTO и CSV при прогретом пайплайне, не включает HTTP-загрузку/браузер и не является проверкой точности на независимых пациентах. Критерий `≤180 с/исследование` пройден **только на этом development-наборе и этой машине**; минимальные аппаратные требования и свободная RAM под нагрузкой не измерены. Старый `local-test-images/benchmark-all.json` относится к прежнему архиву и не описывает текущие веса. Агрегированный результат нового замера сохранён только локально в игнорируемом Git `local-test-images/benchmark-optimized.json`; DICOM и UID не публикуются. Для повторения потребуется каталог с уникальными исследованиями, где 1–3 кадра на Study UID: `python ../scripts/benchmark.py --source ../local-test-images/benchmark-corpus --models models/reference --output ../local-test-images/benchmark-current.json` из `backend/`. Docker startup/healthcheck в текущем окружении не подтверждён.
