FROM python:3.10-slim

RUN apt-get update && apt-get install -y git gcc curl openjdk-17-jdk openjdk-17-jre-headless pkg-config libhdf5-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN pip install h5py poetry==1.8.2 pyspark==3.5.1 wheel==0.43.0 build==1.1.1 && poetry config virtualenvs.create false

WORKDIR /app
RUN git clone https://github.com/google-research/deduplicate-text-datasets.git
WORKDIR /app/deduplicate-text-datasets
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo build

WORKDIR /app
COPY text_dedup /app/text_dedup
COPY pyproject.toml /app
COPY poetry.lock /app
COPY log4j.properties /app
COPY README.md /app/README.md
RUN poetry install
