FROM public.ecr.aws/lambda/python:3.12

COPY ./api ./api
COPY ./data ./data

COPY requirements.txt .

RUN pip3 install -r requirements.txt -U --no-cache-dir

ARG DEFAULT_MODEL
ENV DEFAULT_MODEL=${DEFAULT_MODEL}

CMD [ "api.app.handler" ]
