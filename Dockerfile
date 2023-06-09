#base image
#FROM continuumio/anaconda3
FROM python:3.8.8-slim-buster
WORKDIR /scripts

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY trainingML.py ./trainingML.py
COPY kidney_disease.csv ./kidney_disease.csv

CMD python3 trainingML.py


