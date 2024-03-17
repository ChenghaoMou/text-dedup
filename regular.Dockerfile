FROM python:3.10-slim

RUN apt-get update && apt-get install -y git gcc curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN pip install wheel build poetry && poetry config virtualenvs.create false

WORKDIR /app
RUN git clone https://github.com/google-research/deduplicate-text-datasets.git
WORKDIR /app/deduplicate-text-datasets
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo build

WORKDIR /app
COPY text_dedup /app/text_dedup
COPY pyproject.toml /app
COPY poetry.lock /app
COPY README.md /app/README.md
RUN poetry install
