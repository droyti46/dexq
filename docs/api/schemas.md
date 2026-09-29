# Формат ответа

<p class="lead">Все поля ответа описаны в <code>backend/app/schemas.py</code> (Pydantic). Полная машиночитаемая схема — <a href="http://localhost:8000/openapi.json">/openapi.json</a>.</p>

## AnalysisResult

Результат анализа одного снимка. Пример — реальный ответ сервера на синтетический фантом с наклонённой осью (длинные строки сокращены):

```json
{
  "analysis_id": "an_39d67663332b8d01b9a3",
  "filename": "phantom_spine_02_tilted.dcm",
  "study_uid": "1.2.826.0.1.3680043.8.498.1599…",
  "image_uid": "1.2.826.0.1.3680043.8.498.5430…",
  "anatomical_region": "lumbar_spine",
  "region_source": "model",
  "projection": "unknown",
  "projection_source": "not_determined",
  "needs_review": false,
  "quality_class": 1,
  "violation_types": ["spine_axis"],
  "processing_status": "Success",
  "time_of_processing": 2.3715,
  "checks": [
    {
      "check_id": "spine_axis",
      "title": "Наклон оси позвоночника",
      "status": "failed",
      "violation": true,
      "summary": "Модель отметила нарушение качества.",
      "confidence": null,
      "details": {"score": 8.578},
      "method": "reference_qc",
      "model_status": "reference_onnx_v13"
    }
  ],
  "preview_data_url": "data:image/png;base64,iVBORw0KGgo…",
  "annotated_data_url": "data:image/png;base64,iVBORw0KGgo…",
  "geometry": {
    "image_width": 300,
    "image_height": 320,
    "coordinate_system": "native_pixels_x_right_y_down",
    "axis_line": {
      "top_xy": [129.96, 23.66],
      "bottom_xy": [165.42, 258.72],
      "angle_deg": 8.578
    },
    "vertebral_candidates": [{"center_xy": [129.96, 23.66]}, {"center_xy": [134.95, 74.04]}],
    "gap_lines": [],
    "foreground_bbox": null
  },
  "error": null
}
```

### Итог

| Поле | Тип | Описание |
| --- | --- | --- |
| `processing_status` | `"Success"` \| `"Failure"` | `Success` — все обязательные проверки дали решение |
| `quality_class` | `0` \| `1` \| `null` | `1` — хотя бы одно нарушение, `0` — нарушений нет, `null` при `Failure` |
| `violation_types` | `string[]` | Коды нарушений в алфавитном порядке |
| `needs_review` | `bool` | Модель просит проверить снимок специалистом, например при низкой уверенности роутера |
| `error` | `string` \| `null` | Причина при `Failure` |
| `time_of_processing` | `number` | Время обработки на сервере, секунды |

### Снимок

| Поле | Описание |
| --- | --- |
| `analysis_id` | Идентификатор по содержимому файла: одинаковые файлы получают одинаковый ID |
| `filename` | Имя файла без пути |
| `study_uid`, `image_uid` | StudyInstanceUID и SOPInstanceUID из DICOM; `null` для PNG |
| `anatomical_region` | `lumbar_spine`, `proximal_femur` или `unknown` |
| `region_source` | `model` — определил роутер, `operator` — область передана в запросе |
| `projection` | `AP`, `PA` или `unknown` |
| `projection_source` | `dicom_view_position` — из тега `ViewPosition`, иначе `not_determined` |

### Проверки — `checks[]`

| Поле | Описание |
| --- | --- |
| `check_id` | Код проверки: `spine_positioning`, `spine_axis`, `spine_artifacts`, `hip_positioning_rotation`, `hip_roi` |
| `title` | Название для человека |
| `status` | `passed`, `failed`, `not_evaluated` или `error` |
| `violation` | `true` — нарушение, `false` — норма, `null` — решения нет |
| `summary` | Короткое пояснение |
| `details.score` | Числовая оценка проверки. Для `spine_axis` — угол оси в градусах, для остальных — внутренний скор модели |
| `method`, `model_status` | Источник решения: `reference_qc` и версия комплекта моделей |

В ответ попадают только проверки, относящиеся к области снимка: 3 для позвоночника, 2 для бедра.

### Изображения и геометрия

| Поле | Описание |
| --- | --- |
| `preview_data_url` | Исходный снимок в PNG (data URL) — для показа в интерфейсе |
| `annotated_data_url` | Снимок с нарисованными ориентирами, если модель их вернула |
| `geometry.image_width`, `image_height` | Размер исходного снимка в пикселях |
| `geometry.coordinate_system` | Всегда `native_pixels_x_right_y_down`: пиксели исходного снимка, X вправо, Y вниз |
| `geometry.axis_line` | Ось позвоночника: `top_xy`, `bottom_xy` и угол `angle_deg` |
| `geometry.vertebral_candidates` | Найденные центры позвонков |
| `geometry.gap_lines` | Межпозвоночные промежутки, если найдены |
| `geometry.foreground_bbox` | Для бедра — рамка светлой области снимка. Это **не** анатомическая маска |

## BatchResult

Ответ пакетных методов и `/archive`.

| Поле | Описание |
| --- | --- |
| `items` | Массив [`BatchItem`](#batchitem) в порядке входных файлов |
| `successful` | Сколько файлов получили `Success` |
| `failed` | Сколько файлов получили `Failure` или ошибку чтения |

## BatchItem

| Поле | Описание |
| --- | --- |
| `filename` | Имя файла без пути |
| `input_position` | Порядковый номер во входе, с 1. Различает файлы с одинаковыми именами |
| `result` | [`AnalysisResult`](#analysisresult) или `null`, если файл не удалось даже прочитать |
| `error` | Причина, если `result = null` |

Путь файла внутри ZIP в JSON-ответ `/archive` **не** включается — только в CSV и в поток `archive.stream` (поле `input_path` события `started`).
