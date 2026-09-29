# Офлайн-поставка

<p class="lead">Как собрать комплект, который запускается на машине без интернета, и что из этого уже проверено.</p>

## Что проверено

| Что | Статус |
| --- | --- |
| Установка весов и проверка SHA-256 без сети | <span class="pill ok">проверено</span> |
| `docker compose config` | <span class="pill ok">проверено</span> |
| Сборка и запуск контейнеров, healthcheck, сайт через Nginx, поток ZIP и CSV | <span class="pill ok">проверено</span> 29.09.2026 в VM: Ubuntu Server 24.04.5, Docker Engine 29.8.1, Compose v5.5.1 |
| Полный запуск на чистой машине без сети из сохранённых образов | <span class="pill warn">не проверено</span> |

При проверке образы и зависимости pip/npm скачивались из сети. Флаг `--build` сам по себе не делает сборку автономной: для неё нужны заранее сохранённые образы. Поэтому поставку нельзя называть полностью офлайновой, пока она не проверена на отдельном компьютере без интернета.

## Закреплённые версии

Базовые образы закреплены по multiarch digest, чтобы сборка была воспроизводимой:

| Этап | Образ |
| --- | --- |
| Python | `python:3.12.6-slim-bookworm@sha256:ad48727987b259854d52241fac3bc633574364867b8e20aec305e6e7f4028b26` |
| Сборка frontend | `node:22.23.0-alpine3.23@sha256:35e2f96595091599e7c1fb0b61049e17d8478997b2aa13db51ad7995299fe55a` |
| Nginx | `nginx:1.27.5-alpine3.21@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10` |

Python-зависимости зафиксированы в `backend/requirements.lock`, JavaScript — в `frontend/package-lock.json`. Инференс идёт через `CPUExecutionProvider` ONNX Runtime; работа на GPU не требуется и не проверялась.

Образ backend содержит 74 Python-файла `combined_qc`, побайтово совпадающих с архивом, но **не содержит весов и изображений** — веса монтируются с хоста в `/models/reference` только для чтения.

## Как собрать комплект

<div class="steps" markdown>

1. **Соберите и проверьте образы** на машине с интернетом: `./scripts/run.sh`, затем убедитесь, что сайт и `/api/v1/health` работают.

2. **Сохраните образы в файл:**

    ```bash
    docker save -o submission/dexq-images.tar dexq-backend:latest dexq-frontend:latest
    ```

3. **Упакуйте комплект** из папки `backend/`:

    ```bash
    python ../scripts/package_submission.py \
      --output ../submission/dexq-offline.zip \
      --models models/reference \
      --images ../submission/dexq-images.tar
    ```

    Упаковщик берёт только разрешённые файлы из Git, проверенные веса и сохранённые образы. Папка `data/` и изображения в комплект не попадают.

4. **На целевой машине** распакуйте архив, переложите `models/reference/` в `source/backend/models/reference/`, загрузите образы и запустите:

    ```bash
    docker load -i images/dexq-images.tar
    ./source/scripts/run.sh
    ```

</div>

Структура комплекта:

```text
submission/
  source/                 исходный код и конфигурация без медицинских данных
  source/scripts/run.sh   точка запуска с проверкой SHA-256
  models/reference/       22 файла весов + installed-manifest.json
  images/                 сохранённые Docker-образы
```

!!! danger "Что нельзя публиковать"
    Не загружайте в Git и внешние сервисы папку `submission/`, DICOM, PNG-превью, файлы весов ONNX/NPZ и построчные метрики. Веса и архивы передавайте только по согласованному локальному каналу.

## Ресурсы и время

Замер на ноутбуке с Intel Core i7-1355U, 16 ГБ ОЗУ, Windows 11, Python 3.12.6, `CPUExecutionProvider`:

| Показатель | Значение |
| --- | --- |
| Набор | 255 уникальных DICOM из 102 исследований, 1–3 снимка на исследование |
| Успешно обработано | 255 из 255 |
| Среднее время на исследование | **4,26 с** |
| Максимум на исследование | **15,9 с** (норматив задания — ≤ 180 с) |
| Холодная загрузка моделей | 2,65 с |
| Веса на диске | 22 файла, ~264 МиБ |

Время включает чтение файла, декодирование, инференс и формирование ответа при уже загруженных моделях. Загрузка по HTTP и отрисовка в браузере не учитываются. Это development-набор и одна машина, поэтому замер не заменяет проверку минимальных аппаратных требований.

Повторить замер (из папки `backend/`, нужна папка с уникальными исследованиями):

```bash
python ../scripts/benchmark.py \
  --source ../local-test-images/benchmark-corpus \
  --models models/reference \
  --output ../local-test-images/benchmark-current.json
```
