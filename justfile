# Set default recipe to run when just is called without arguments
default: help

# ⚙️ Install the virtual environment and pre-commit hooks
install:
    @echo "⚙️ Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

format:
    @echo "👕️ Formatting code with ruff"
    uv run ruff format .

# 🔍 Run code quality tools (linting, type checking, dependency checks)
check:
    @echo "🔍 Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    @echo "🔍 Linting code: Running pre-commit"
    uv run pre-commit run -a
    @echo "🔍 Static type checking: Running mypy"
    uv run mypy
    @echo "🔍 Checking for obsolete dependencies: Running deptry"
    uv run deptry src

# 🧪 Test the code with pytest and coverage
test:
    @echo "🧪 Testing code: Running pytest"
    uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml --cov-report=term-missing

# 📊 Run the gradio app for report visualization
report:
    @echo "📊 Running gradio app"
    uv run python -m text_dedup.utils.gradio.run

# 🧹 Clean build artifacts
clean-build:
    @echo "🧹 Removing build artifacts"
    rm -rf ./dist

clean: clean-build
    @echo "🧹 Cleaning cache and build artifacts"
    find . -name "*cache*" -type d -exec rm -rf {} +
    rm -rf coverage.xml

# 🏗️ Build wheel file
build: clean-build
    @echo "🏗️ Creating wheel file"
    uvx --from build pyproject-build --installer uv

# 📤 Publish release to PyPI
publish:
    @echo "📤 Publishing to PyPI"
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# 🚢 Build and publish in one step
build-and-publish: build publish

# 💡 Show this help message
help:
    @just --list --unsorted

run:
    @echo "🏃️ Run with the config file"
    uv run --frozen -m text_dedup.minhash

app:
    @echo "🏃️ Run gradio app"
    uv run --with gradio,plotly,gradio_rangeslider -m text_dedup.utils.gradio.run
