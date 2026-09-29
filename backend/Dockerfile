FROM python:3.12.6-slim-bookworm@sha256:ad48727987b259854d52241fac3bc633574364867b8e20aec305e6e7f4028b26

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY app ./app
COPY reference ./reference
RUN pip install --upgrade pip==26.0.1 setuptools==80.9.0 \
    && pip install -r requirements.lock \
    && pip install --no-build-isolation --no-deps . \
    && useradd --create-home --uid 10001 dexq
USER dexq

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
