FROM python:3.12-slim

WORKDIR /pitchcraft-model

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 \
    && pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY model_server ./model_server

COPY model_shared ./model_shared

COPY rnn_support_models ./rnn_support_models

COPY pitch_arsenal ./pitch_arsenal

COPY feature_list feature_list

COPY certs ./certs

CMD ["sh", "-c", "python -m model_shared.setup && exec uvicorn model_server.src.api:app --host 0.0.0.0 --port 8000 --workers 4"]
