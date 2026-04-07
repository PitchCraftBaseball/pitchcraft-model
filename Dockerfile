FROM python:3.12

WORKDIR /pitchcraft-model

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY model_server ./model_server

COPY model_shared ./model_shared

COPY certs ./certs

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "model_server.src.api:app", "--host", "0.0.0.0", "--port", "8000"]