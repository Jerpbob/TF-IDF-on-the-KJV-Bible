FROM quay.io/jupyter/scipy-notebook

WORKDIR /app/data

COPY /data .

WORKDIR /app