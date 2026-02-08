FROM python:3.12

WORKDIR /pitchcraft-model

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY /model_server ./model_server

RUN ["python", "./model_server/build_model_config.py"] 

CMD ["uvicorn", "model_server.api:app", "--host", "0.0.0.0", "--port", "8000"]