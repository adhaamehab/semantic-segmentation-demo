FROM ubuntu:16.04


RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

Run pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . /app

RUN mkdir models

RUN wget -O models/ade20kmodel_may2019.gz https://www.dropbox.com/s/upy0a07e38243wc/ade20kmodel_may2019.gz?dl=0

EXPOSE 8050

RUN python3 app.py
