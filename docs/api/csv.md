# CSV и командная строка

<p class="lead">Итоговая таблица в формате задания — одна строка на каждый входной снимок. Её выдают методы <code>/batch.csv</code> и <code>/archive.csv</code>, командная строка и пункт сайта «Для организаторов · CSV».</p>

## Формат

```text
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing
phantom_spine_01.dcm,1.2.826.0.1…,1.2.826.0.1…,lumbar_spine,0,,Success,3.5297
phantom_hip_left.dcm,1.2.826.0.1…,1.2.826.0.1…,proximal_femur,1,hip_roi,Success,0.7037
damaged_file.dcm,,,unknown,,,Failure,0.0001
```

| Столбец | Значение |
| --- | --- |
| `path_to_study` | Путь файла: относительный путь внутри ZIP или папки, для пакета — имя файла |
| `study_uid` | StudyInstanceUID |
| `image_uid` | SOPInstanceUID |
| `anatomical_region` | `lumbar_spine`, `proximal_femur` или `unknown` |
| `quality_class` | `0` или `1`; пусто при `Failure` |
| `violation_type` | Коды нарушений через `;` в алфавитном порядке; пусто, если нарушений нет или `Failure` |
| `processing_status` | `Success` или `Failure` |
| `time_of_processing` | Секунды |

Файл в UTF-8 с BOM, разделитель — запятая. Путь, похожий на формулу Excel (начинается с `=`, `+`, `-`, `@`), получает ведущий апостроф.

## Командная строка

Для обработки набора без сервера и браузера — из папки `backend/`:

```bash
.venv/bin/python -m app.cli ВХОД ВЫХОД.csv [--region auto|lumbar_spine|proximal_femur] [--max-file-mb 50]
```

`ВХОД` может быть:

- ZIP-архивом — пути в CSV будут относительными внутри архива;
- папкой — файлы ищутся рекурсивно, пути относительно папки, порядок алфавитный;
- одним файлом `.dcm`, `.dicom` или `.png`.

Пример:

```bash
cd backend
.venv/bin/python -m app.cli ~/data/test_set.zip ~/results/test_set.csv
```

```text
Обработано: 998; ошибок: 2; CSV: /home/user/results/test_set.csv
```

После `pip install -e .` та же команда доступна как `dexq-batch`:

```bash
dexq-batch ~/data/test_set.zip ~/results/test_set.csv
```

!!! tip "Какой способ выбрать для тестового набора"
    - Сервер уже запущен в Docker — `POST /api/v1/analyses/archive.csv`.
    - Docker нет, но есть Python — CLI: модели загружаются один раз, без HTTP-накладных расходов.
    - Нужно просмотреть снимки глазами — сайт и «Файл → Для организаторов · CSV».
