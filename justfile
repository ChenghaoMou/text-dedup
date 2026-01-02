default: help

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set dotenv-load := true

uvr := "uv run"
uvx := "uvx"

install:
    @echo "Creating virtual environment using uv"
    uv sync
    {{uvr}} pre-commit install

format:
    @echo "Formatting code with ruff"
    {{uvr}} ruff format .

check:
    @echo "Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    @echo "Linting code: Running pre-commit"
    {{uvr}} pre-commit run -a
    @echo "Static type checking: Running mypy"
    {{uvr}} mypy src report
    @echo "Checking for obsolete dependencies: Running deptry"
    {{uvr}} deptry src

test:
    @echo "Testing code: Running pytest"
    {{uvr}} python -m pytest --doctest-modules --ignore=third_party --cov --cov-config=pyproject.toml --cov-report=xml --cov-report=term-missing

tox:
    @echo "Running tox across Python 3.12, 3.13"
    {{uvx}} --with tox-uv tox

benchmark dataset alg="minhash,simhash":
    {{uvr}} python -m benchmarks.run_benchmark --dataset {{dataset}} --algorithms {{alg}}

benchmark-all: benchmark-core benchmark-news
    @echo "All benchmarks completed"

benchmark-core:
    @just benchmark core

benchmark-news:
    @just benchmark news

report:
    @echo "Running gradio app"
    uv run --with gradio,plotly,gradio_rangeslider -m report.run

clean-build:
    @echo "Removing build artifacts"
    rm -rf ./dist

clean: clean-build
    @echo "Removing cache and artifacts"
    rm -rf .ruff_cache .mypy_cache .pytest_cache
    find . -type d -name __pycache__ -exec rm -r {} +

build: clean-build
    @echo "Creating wheel file"
    {{uvx}} --from build pyproject-build --installer uv

publish:
    @echo "Publishing to PyPI"
    {{uvx}} twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

build-and-publish: build publish

help:
    @just --list --unsorted

run:
    @echo "Run with the config file"
    uv run --frozen -m text_dedup.minhash

app:
    @echo "Run gradio app"
    uv run --with gradio,plotly,gradio_rangeslider -m text_dedup.utils.gradio.run
