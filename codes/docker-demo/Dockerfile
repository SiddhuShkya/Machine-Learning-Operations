## Use a lightweight Python 3.8 image based on Alpine Linux as the base image
FROM python:3.8-alpine

## Copy all files from the current directory on the host to /app inside the container
COPY . /app

## Set /app as the working directory for all subsequent instructions
WORKDIR /app 

## Install Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt

## Specify the default command to run when the container starts
CMD ["python", "app.py"]
