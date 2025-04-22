FROM public.ecr.aws/lambda/python:3.12

# Copy the contents of the api directory to the root of the container
COPY ./api/* .

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip3 install -r requirements.txt -U --no-cache-dir

# Set the Lambda handler
CMD [ "app.handler" ]