#base image
#FROM continuumio/anaconda3
FROM python:3.8.8-slim-buster

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY trainingML.ipynb ./trainingML.ipynb

CMD python3 trainingML.ipynb 


