# Запуск DEXQ

<p class="lead">DEXQ передаётся одним архивом: в нём код, веса моделей и конфигурация Docker. Ничего скачивать отдельно не нужно — достаточно распаковать архив и запустить одну команду.</p>

## За три шага

<div class="steps" markdown>

1. **Распакуйте архив** в любую папку и перейдите в неё:

    ```bash
    unzip dexq.zip
    cd dexq
    ```

2. **Запустите DEXQ в Docker:**

    === "Linux / macOS"

        ```bash
        ./scripts/run.sh
        ```

    === "Windows (PowerShell)"

        ```powershell
        $env:DEXQ_MODELS_DIR = "$PWD\backend\models\reference"
        docker compose up --build
        ```

3. **Откройте сайт** <http://localhost:3000>. Интерактивная документация API — <http://localhost:8000/docs>.

</div>

!!! tip "Первый запуск дольше обычного"
    Docker собирает образы и ставит зависимости, а сервер при старте загружает модели в память. Сайт станет доступен, когда проверка `GET /api/v1/health` ответит `ready`.

## Что внутри архива

```text
dexq/
├── backend/
│   ├── app/                  код сервера
│   ├── reference/            модельное ядро
│   └── models/reference/     веса моделей — уже на месте
├── frontend/                 сайт
├── scripts/run.sh            проверка весов и запуск Docker
├── docker-compose.yml
└── docs/                     эта документация
```

Веса лежат в `backend/models/reference/` вместе с манифестом контрольных сумм. При запуске они проверяются автоматически — подробнее на странице [Веса моделей](weights.md).

## Какой способ выбрать

<div class="grid cards cards-2" markdown>

-   :material-docker:{ .lg .middle } **В Docker**

    ---

    Основной способ. Нужен только Docker с Compose. Сервер, сайт и Nginx поднимаются одной командой.

    [:octicons-arrow-right-24: Запуск в Docker](docker.md)

-   :material-console:{ .lg .middle } **Без Docker**

    ---

    Для разработки или если Docker недоступен. Нужны Python 3.12 и Node.js 22.

    [:octicons-arrow-right-24: Запуск без Docker](local.md)

</div>

## Требования

| Что | Для чего |
| --- | --- |
| Docker Engine + Compose (или Docker Desktop) | Запуск в Docker |
| Python 3 | Предварительная проверка весов в `run.sh` — подойдёт любой Python 3 из системы |
| 4 ГБ ОЗУ, ~2 ГБ на диске | Модели (~264 МиБ), образы и временные файлы |

Видеокарта не нужна: все модели работают на процессоре.

Если какой-то шаг не проходит, загляните в раздел [Если что-то не так](troubleshooting.md).
