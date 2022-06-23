FROM tensorflow/tensorflow:latest

RUN apt-get update
RUN pip install --upgrade pip && pip install flask gunicorn numpy

WORKDIR /root/app
RUN mkdir ./static
COPY . .

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 predictor:app