FROM python:3.10-slim

RUN apt-get update && apt-get install -y openjdk-17-jdk openjdk-17-jre-headless gcc
RUN pip install poetry && poetry config virtualenvs.create false
RUN pip install pyspark

WORKDIR /app
COPY text_dedup /app/text_dedup
COPY pyproject.toml /app
COPY poetry.lock /app
COPY log4j.properties /app
COPY README.md /app/README.md
RUN poetry install
