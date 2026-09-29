# Эндпоинты

Все пути ниже — относительно `http://localhost:8000/api/v1`.

## <span class="http get">GET</span> `/health` { #health }

Проверяет, что веса установлены, их контрольные суммы совпадают и модели загружаются. Первый вызов после старта сервера загружает модели в память и занимает несколько секунд.

=== "Готов — 200"

    ```json
    {"status": "ready", "service": "DEXQ API", "version": "0.1.0"}
    ```

=== "Не готов — 503"

    ```json
    {"detail": "Локальный комплект моделей недоступен"}
    ```

Этот же адрес использует healthcheck Docker Compose.

## <span class="http get">GET</span> `/checks` { #checks }

Список проверок, которые умеет выполнять сервер.

```json
[
  {"check_id": "spine_positioning", "title": "Охват позвоночника",
   "regions": ["lumbar_spine"], "model_status": "reference_onnx_v13"},
  {"check_id": "spine_axis", "title": "Наклон оси позвоночника",
   "regions": ["lumbar_spine"], "model_status": "reference_onnx_v13"},
  {"check_id": "spine_artifacts", "title": "Артефакты позвоночника",
   "regions": ["lumbar_spine"], "model_status": "reference_onnx_v13"},
  {"check_id": "hip_positioning_rotation", "title": "Позиционирование / ротация бедра",
   "regions": ["proximal_femur"], "model_status": "reference_onnx_v13"},
  {"check_id": "hip_roi", "title": "Охват ROI бедра",
   "regions": ["proximal_femur"], "model_status": "reference_onnx_v13"}
]
```

## <span class="http post">POST</span> `/analyses` { #analyses }

Анализ одного снимка. Ответ — объект [`AnalysisResult`](schemas.md#analysisresult).

| Поле формы | Обязательно | Описание |
| --- | --- | --- |
| `file` | да | DICOM (`.dcm`, `.dicom`) или PNG в оттенках серого |
| `anatomical_region` | нет | `auto`, `lumbar_spine` или `proximal_femur` |

=== "curl"

    ```bash
    curl -F "file=@study.dcm" \
         -F "anatomical_region=auto" \
         http://localhost:8000/api/v1/analyses
    ```

=== "Python"

    ```python
    import httpx

    with open("study.dcm", "rb") as f:
        response = httpx.post(
            "http://localhost:8000/api/v1/analyses",
            files={"file": ("study.dcm", f, "application/dicom")},
            data={"anatomical_region": "auto"},
            timeout=120,
        )
    response.raise_for_status()
    result = response.json()
    print(result["quality_class"], result["violation_types"])
    ```

=== "JavaScript"

    ```js
    const form = new FormData();
    form.append("file", fileInput.files[0]);
    form.append("anatomical_region", "auto");

    const response = await fetch("/api/v1/analyses", { method: "POST", body: form });
    if (!response.ok) throw new Error((await response.json()).detail);
    const result = await response.json();
    ```

Если снимок не удалось обработать (`processing_status = Failure`), этот метод отвечает **422** с причиной в `detail`, а не объектом результата:

```json
{"detail": "Не удалось безопасно обработать файл или модель недоступна"}
```

## <span class="http post">POST</span> `/analyses/batch` { #batch }

Пакет из 1–200 файлов за один запрос. Ответ — [`BatchResult`](schemas.md#batchresult): по строке на каждый файл в исходном порядке.

```bash
curl -F "files=@a.dcm" -F "files=@b.dcm" -F "files=@c.dcm" \
     http://localhost:8000/api/v1/analyses/batch
```

```json
{
  "items": [
    {"filename": "a.dcm", "input_position": 1, "result": {"processing_status": "Success", "...": "..."}, "error": null},
    {"filename": "b.dcm", "input_position": 2, "result": {"processing_status": "Failure", "...": "..."}, "error": null},
    {"filename": "c.dcm", "input_position": 3, "result": null, "error": "Файл превышает ограничение 50 МБ"}
  ],
  "successful": 1,
  "failed": 2
}
```

В отличие от `/analyses`, ошибка одного файла **не** превращает весь ответ в 422: она попадает в строку этого файла.

## <span class="http post">POST</span> `/analyses/batch.csv` { #batch-csv }

То же, что `/batch`, но ответ — CSV-файл `dexq-results.csv` в формате задания. Описание столбцов — в разделе [CSV и CLI](csv.md).

```bash
curl -F "files=@a.dcm" -F "files=@b.dcm" \
     -o results.csv \
     http://localhost:8000/api/v1/analyses/batch.csv
```

## <span class="http post">POST</span> `/analyses/archive` { #archive }

ZIP-архив до 1000 снимков. Сервер распаковывает архив в память (на диск ничего не пишется) и возвращает [`BatchResult`](schemas.md#batchresult).

| Поле формы | Обязательно | Описание |
| --- | --- | --- |
| `archive` | да | ZIP с DICOM или PNG, можно с вложенными папками |
| `anatomical_region` | нет | Как в `/analyses` |

```bash
curl -F "archive=@studies.zip" http://localhost:8000/api/v1/analyses/archive
```

Архив отклоняется целиком (422), если:

- он повреждён, зашифрован или сжат неподдерживаемым методом;
- в нём нет изображений или их больше 1000;
- отдельный файл больше 50 МБ или всё вместе после распаковки больше 2 ГБ;
- имена файлов повторяются (без учёта регистра);
- пути небезопасны: `..`, абсолютные пути, `\`, `:`, символьные ссылки или зарезервированные имена Windows (`CON`, `NUL`, `COM1`…).

## <span class="http post">POST</span> `/analyses/archive.csv` { #archive-csv }

То же, что `/archive`, но ответ — CSV `dexq-archive-results.csv`. В `path_to_study` записывается относительный путь файла внутри архива. **Этот метод подходит для проверки на тестовом наборе.**

```bash
curl -F "archive=@studies.zip" -o results.csv \
     http://localhost:8000/api/v1/analyses/archive.csv
```

## <span class="http post">POST</span> `/analyses/archive.stream` { #archive-stream }

ZIP-архив с результатами **по мере готовности**. Ответ — `application/x-ndjson`: каждая строка — отдельный JSON-объект. Так сайт показывает прогресс, не дожидаясь конца всего архива.

| `type` | Поля | Когда приходит |
| --- | --- | --- |
| `started` | `filename`, `input_path`, `input_position` | Началась обработка очередного файла |
| `result` | `item` — объект [`BatchItem`](schemas.md#batchitem) | Файл обработан |
| `complete` | `successful`, `failed` | Архив обработан полностью |
| `error` | `detail` | Архив нельзя обрабатывать дальше |

```text
{"type": "started", "filename": "img_1.dcm", "input_path": "study_01/img_1.dcm", "input_position": 1}
{"type": "result", "item": {"filename": "img_1.dcm", "input_position": 1, "result": {...}, "error": null}}
{"type": "complete", "successful": 1, "failed": 0}
```

Если поток оборвался без события `complete`, считайте обработку незавершённой.

Пример чтения потока на Python:

```python
import json
import httpx

with open("studies.zip", "rb") as f, httpx.stream(
    "POST",
    "http://localhost:8000/api/v1/analyses/archive.stream",
    files={"archive": ("studies.zip", f, "application/zip")},
    timeout=None,
) as response:
    for line in response.iter_lines():
        event = json.loads(line)
        if event["type"] == "result":
            result = event["item"]["result"] or {}
            print(event["item"]["filename"], result.get("quality_class"))
```

## <span class="http get">GET</span> `/` (вне `/api/v1`)

Служебный ответ корня сервера:

```json
{"title": "DEXQ API", "docs": "/docs", "status": "ok"}
```
