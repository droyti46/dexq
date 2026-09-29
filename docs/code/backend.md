# Backend

<p class="lead">Сервер анализа — пакет <code>app</code> в папке <code>backend/</code>. Он принимает файлы по HTTP, вызывает модельное ядро и возвращает результат в едином формате.</p>

## Модули `backend/app`

| Файл | Ответственность |
| --- | --- |
| `main.py` | Создаёт приложение FastAPI: один общий `Analyzer`, лимиты из переменных окружения, CORS для `localhost:3000` и `:5173`, корневой `/` |
| `api.py` | Все HTTP-маршруты `/api/v1/*`: чтение загрузок с ограничением размера, вызов анализа в пуле потоков, преобразование ошибок в 422/503, NDJSON-поток |
| `analyzer.py` | `Analyzer` — сердце сервиса: читает безопасные DICOM-поля, вызывает runtime, собирает проверки, итог и геометрию в `AnalysisResult` |
| `model_runtime.py` | `ReferenceRuntime` — единственный экземпляр `QCPipeline`: ленивая загрузка с проверкой весов, блокировка, временный файл, фильтр полей ответа |
| `checks/base.py` | Контракт `QualityCheck` и контекст `AnalysisContext` |
| `checks/reference.py` | `ReferenceCheck` и список `REFERENCE_CHECKS` — пять проверок, которые переводят метки эталона в `CheckResult` |
| `imaging.py` | Чтение DICOM/PNG: `safe_dicom_fields` (только UID и ViewPosition), нормализация пикселей, превью, `content_id` |
| `batch.py` | Проверка и чтение ZIP в памяти (`iter_archive`), пакетный анализ (`analyze_items`), CSV (`render_csv`), сбор файлов для CLI |
| `schemas.py` | Pydantic-модели ответа: `AnalysisResult`, `CheckResult`, `ImageGeometry`, `BatchItem`, `BatchResult`, `CheckInfo` |
| `reference_assets.py` | Установка весов из архива (`prepare_reference`) и проверка SHA-256 по манифесту (`verify_models`) |
| `cli.py` | Командная строка `python -m app.cli` / `dexq-batch` |

!!! note "Модули `checks/spine.py`, `checks/hip.py`, `checks/artifact.py`"
    Это ранние эвристические проверки, написанные до подключения эталонной модели. `Analyzer` их **не использует** — рабочий список проверок задаёт только `REFERENCE_CHECKS`.

## Модельное ядро `backend/reference/combined_qc`

Код эталонного решения, скопированный из архива без изменений (74 файла). Главный вход — `combined_qc.router.QCPipeline`:

1. **Роутер** (ResNet18 + логистическая регрессия) по пикселям решает, позвоночник это или бедро. Если уверенность ниже 0,9, снимок помечается `needs_review`.
2. **Позвоночник:** эвристика охвата, SpineNet для центров позвонков и угла оси, классификатор артефактов.
3. **Бедро:** модель стороны, затем два классификатора поверх признаков ResNet18 — позиционирование/ротация и охват ROI.

Все нейросети — ONNX, запускаются на `CPUExecutionProvider`. Подробно о предобработке и порогах — в разделе [Модель](../model/index.md).

## Как устроен `ReferenceRuntime`

- **Одна загрузка на процесс.** Модели загружаются при первом `ready()` или `infer()` и дальше переиспользуются.
- **Проверка весов перед загрузкой.** `verify_models` сверяет каждый файл с `installed-manifest.json`; при расхождении — `ValueError`, `/health` отвечает 503.
- **Последовательные вызовы.** `QCPipeline` не рассчитан на параллельный доступ, поэтому вызовы защищены `RLock`.
- **Временный файл.** Эталон принимает путь, а не байты, поэтому файл записывается во временную папку и удаляется сразу после вызова.
- **Выбор области вручную.** При `anatomical_region = lumbar_spine | proximal_femur` роутер пропускается и вызывается профиль нужной области с уже загруженными весами; `region_source = "operator"`.
- **Белый список полей.** Наружу передаются только известные поля (`labels`, `scores`, `geometry`, `image`…), без путей и служебных данных.

## Как `Analyzer` выставляет итог

```python
checks = [check.run(raw) for check in self.checks if region in check.regions]
violations = [item.check_id for item in checks if item.violation is True]
if checks and all(item.violation is not None for item in checks):
    quality_class = int(bool(violations))           # 1 — хотя бы одно нарушение
    if raw.get("any_violation") != bool(quality_class):
        raise ValueError("Решение модели противоречит независимым проверкам")
    processing_status = "Success"
else:
    error_message = "Не удалось определить все обязательные проверки качества"
```

- `Success` — только если **все** проверки области дали определённое решение.
- Итог дополнительно сверяется с собственным флагом эталона `any_violation` — расхождение считается ошибкой, а не поводом выбрать одно из двух.
- Любое исключение (`ValueError`, `OSError`, `KeyError`, `TypeError`, `RuntimeError`) превращается в `Failure` с нейтральным текстом — без подробностей, которые могли бы раскрыть содержимое файла.

## Как добавить проверку

1. Убедитесь, что модельное ядро возвращает для неё метку в `labels` и, по возможности, скор в `scores`.
2. Добавьте строку в `REFERENCE_CHECKS` (`backend/app/checks/reference.py`): код, понятное название и область.
3. Сайт показывает проверку по её `title` из ответа, поэтому отдельно регистрировать её в интерфейсе не нужно. Если для неё нужны свои ориентиры на снимке, расширьте `ImageGeometry` и `StudyViewer`.
4. Обновите контрактные тесты `backend/tests/test_analysis_contract.py` и `test_api.py`.
5. Зафиксируйте в [описании модели](../model/index.md) версию весов, SHA-256, предобработку и порог.

Новая проверка должна реализовывать контракт `QualityCheck`: модель и её предобработка остаются внутри проверки, наружу выходит только `CheckResult`.

## Настройки окружения

| Переменная | По умолчанию | Где используется |
| --- | --- | --- |
| `DEXQ_MODELS_DIR` | `backend/models/reference` | `ReferenceRuntime` |
| `DEXQ_MAX_UPLOAD_MB` | `50` | `main.py` → лимит одного файла |
| `DEXQ_MAX_ARCHIVE_MB` | `1024` | `main.py` → лимит ZIP |

## Скрипты `scripts/`

| Скрипт | Что делает |
| --- | --- |
| `prepare_reference.py` | Установка весов из архива с проверкой SHA-256 |
| `run.sh` | Проверка весов и `docker compose up --build` |
| `prepare_samples.py` | Раскладывает локальные примеры из архива по экспертным классам в `local-test-images/` (папка не попадает в Git) |
| `benchmark.py` | Замер времени на исследование |
| `metrics.py` | Метрики и bootstrap-интервалы по сохранённым решениям эталона |
| `package_submission.py` | Сборка офлайн-комплекта |
| `motion_demo.py`, `motion_serious.py` | Генерация демо-видео для лендинга |
