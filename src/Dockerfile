FROM public.ecr.aws/lambda/python:3.12

# Add Lambda Web Adapter for API Gateway response streaming
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

COPY ./api ./api

COPY requirements.txt .

RUN pip3 install -r requirements.txt -U --no-cache-dir

# Preload tiktoken encoding: https://github.com/aws-samples/bedrock-access-gateway/issues/118
ENV TIKTOKEN_CACHE_DIR=/var/task/.cache/tiktoken
RUN python3 -c 'import tiktoken_ext.openai_public as tke; tke.cl100k_base()'

# Lambda Web Adapter requires overriding the Lambda base image entrypoint
# to run the web app directly instead of the Lambda runtime handler
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]