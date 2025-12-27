# Set default recipe to run when just is called without arguments
default: help

# Install the virtual environment and pre-commit hooks
install:
    @echo "Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

format:
    @echo "Formatting code with ruff"
    uv run ruff format .

# Run code quality tools (linting, type checking, dependency checks)
check:
    @echo "Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    @echo "Linting code: Running pre-commit"
    uv run pre-commit run -a
    @echo "Static type checking: Running mypy"
    uv run mypy src report
    @echo "Checking for obsolete dependencies: Running deptry"
    uv run deptry src

# Test the code with pytest and coverage
test:
    @echo "Testing code: Running pytest"
    uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml --cov-report=term-missing

# Run tests across multiple Python versions using tox
tox:
    @echo "Running tox across Python 3.12, 3.13"
    uvx --with tox-uv tox

# Run benchmarks
# Run all benchmarks
benchmark-all: benchmark-core benchmark-news
    @echo "All benchmarks completed"

# Run CORE dataset benchmarks (MinHash + SimHash)
benchmark-core:
    uv run python -m benchmarks.run_benchmark --dataset core --algorithms minhash,simhash

# Run CORE dataset benchmark with MinHash only
benchmark-core-minhash:
    uv run python -m benchmarks.run_benchmark --dataset core --algorithms minhash

# Run CORE dataset benchmark with SimHash only
benchmark-core-simhash:
    uv run python -m benchmarks.run_benchmark --dataset core --algorithms simhash

# Run NEWS-COPY dataset benchmarks (MinHash + SimHash)
benchmark-news:
    uv run python -m benchmarks.run_benchmark --dataset news --algorithms minhash,simhash

# Run NEWS-COPY dataset benchmark with MinHash only
benchmark-news-minhash:
    uv run python -m benchmarks.run_benchmark --dataset news --algorithms minhash

# Run NEWS-COPY dataset benchmark with SimHash only
benchmark-news-simhash:
    uv run python -m benchmarks.run_benchmark --dataset news --algorithms simhash

# Run the gradio app for report visualization
report:
    @echo "Running gradio app"
    uv run --with gradio,plotly,gradio_rangeslider -m report.run

# Clean build artifacts
clean-build:
    @echo "Removing build artifacts"
    rm -rf ./dist

clean: clean-build
    @echo "Removing cache and artifacts"
    rm -rf .ruff_cache .mypy_cache .pytest_cache
    find . -type d -name __pycache__ -exec rm -r {} +

# Build wheel file
build: clean-build
    @echo "Creating wheel file"
    uvx --from build pyproject-build --installer uv

# Publish release to PyPI
publish:
    @echo "Publishing to PyPI"
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# Build and publish in one step
build-and-publish: build publish

# Show this help message
help:
    @just --list --unsorted

run:
    @echo "Run with the config file"
    uv run --frozen -m text_dedup.minhash

app:
    @echo "Run gradio app"
    uv run --with gradio,plotly,gradio_rangeslider -m text_dedup.utils.gradio.run
