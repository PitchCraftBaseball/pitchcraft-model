FROM python:3.12-slim

WORKDIR /pitchcraft-model

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 \
    && pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY model_server ./model_server

COPY model_shared ./model_shared

COPY certs ./certs

CMD ["uvicorn", "model_server.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
