FROM python:3.12.6-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY app ./app
RUN pip install --upgrade pip==26.0.1 \
    && pip install -r requirements.lock \
    && pip install --no-deps .

RUN useradd --create-home --uid 10001 dexq
USER dexq

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
