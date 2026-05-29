# syntax=docker/dockerfile:1.7

# ---- builder: install deps into a relocatable prefix ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-compile --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0 -r requirements.txt \
 && find /install -type d -name __pycache__ -exec rm -rf {} + \
 && find /install -type d -name tests -exec rm -rf {} + \
 && find /install -name '*.pyc' -delete \
 && find /install -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true

# ---- runtime: fresh slim base, just the install tree + app code ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /pitchcraft-model

COPY --from=builder /install /usr/local

COPY model_server ./model_server
COPY model_shared ./model_shared
COPY rnn_support_models ./rnn_support_models
COPY pitch_arsenal ./pitch_arsenal
COPY feature_list ./feature_list
COPY certs ./certs

CMD ["sh", "-c", "python -m model_shared.setup && exec uvicorn model_server.src.api:app --host 0.0.0.0 --port 8000"]
