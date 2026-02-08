FROM python:3.12

WORKDIR /pitchcraft-model

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY /model_development ./model_development

RUN ["python", "./model_development/build_model_config.py"] 

CMD ["uvicorn", "model_development.api:app", "--host", "0.0.0.0", "--port", "8000"]