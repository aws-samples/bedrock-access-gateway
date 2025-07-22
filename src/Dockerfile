FROM python:3.12-slim

COPY ./api ./api

COPY requirements.txt .

RUN pip3 install -r requirements.txt -U --no-cache-dir

CMD [ "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000" ]