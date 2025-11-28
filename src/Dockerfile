FROM public.ecr.aws/lambda/python:3.12

# Add Lambda Web Adapter for API Gateway response streaming
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

COPY ./api ./api

COPY requirements.txt .

RUN pip3 install -r requirements.txt -U --no-cache-dir

# Lambda Web Adapter requires overriding the Lambda base image entrypoint
# to run the web app directly instead of the Lambda runtime handler
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]