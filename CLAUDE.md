# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

text-dedup is a Python library for text deduplication using various algorithms, primarily MinHash. It's designed to handle large-scale text datasets by identifying and removing duplicate content efficiently.

## Development Commands

The project uses both `just` (primary) and legacy `make` commands:

### Primary Commands (just)
- `just install` - Set up virtual environment and pre-commit hooks using uv
- `just install-rust` - Install with Rust extensions for optimal performance
- `just check` - Run all code quality checks (linting, type checking, dependency checks)
- `just test` - Run pytest with coverage reporting
- `just test-rust` - Test Rust extensions and performance comparison
- `just format` - Format code with ruff
- `just build` - Build wheel package
- `just build-rust` - Build/rebuild Rust extensions only
- `just clean` - Clean build artifacts and cache
- `just clean-rust` - Clean Rust build artifacts
- `just run` - Run deduplication with config.toml
- `just app` - Run Gradio reporting app
- `just report` - Run Gradio visualization app

### Testing and Quality
- `uv run python -m pytest tests --cov --cov-config=pyproject.toml --cov-report=xml` - Run tests with coverage
- `uv run mypy` - Type checking
- `uv run ruff format .` - Format code
- `uv run pre-commit run -a` - Run all pre-commit hooks
- `uv run deptry src` - Check for obsolete dependencies

### Running the Application
- `uv run -m text_dedup.minhash` - Run MinHash deduplication with config.toml
- `uv run -m text_dedup.utils.gradio.run` - Launch Gradio reporting interface

### Building Rust Extensions
- `uv run maturin develop --manifest-path third_party/uf_rush/Cargo.toml` - Build and install Rust Union-Find extension
- The Rust extension provides significant performance improvements for large datasets
- Extension automatically used when available; falls back to Python implementation

## Code Architecture

### Core Components
- **Main Algorithm**: `src/text_dedup/minhash.py` - Primary MinHash implementation with clustering
- **Configuration System**: `src/text_dedup/config/` - Pydantic-based config management
  - `base.py` - Main Config class reading from config.toml
  - `algorithms.py` - Algorithm-specific configurations (MinHash, SimHash, etc.)
  - `input_configs.py` - Data source configurations
  - `output_configs.py` - Output format configurations

### Key Utilities
- **Hash Functions**: `src/text_dedup/utils/hashfunc.py` - SHA1 and XXH3 implementations
- **Union Find**: `src/text_dedup/utils/union_find.py` - Original Python implementation
- **Fast Union Find**: `src/text_dedup/utils/union_find_rust.py` - Rust-backed implementation for performance
- **Tokenization**: `src/text_dedup/utils/tokenization.py` - N-gram generation
- **Jaccard Similarity**: `src/text_dedup/utils/jaccard.py` - False positive verification
- **Data I/O**: `src/text_dedup/data_sources/io.py` - Dataset loading/saving
- **Gradio App**: `src/text_dedup/utils/gradio/` - Web interface for result visualization

### Configuration
- Primary config file: `config.toml` - TOML-based configuration for input, algorithm, and output settings
- Algorithm supports MinHash with configurable parameters (hash_bits, num_perm, threshold, etc.)
- Supports HuggingFace datasets as input with various output formats

### Key Features
- **MinHash LSH**: Primary deduplication algorithm with configurable bands/rows
- **False Positive Checking**: Optional pairwise Jaccard verification for precision
- **Multi-processing**: Configurable parallel processing for large datasets
- **Memory Optimization**: Reference counting disabled for Union Find efficiency
- **Rust Integration**: Fast Union-Find implementation with Python bindings (`third_party/uf_rush`)
- **Automatic Performance Optimization**: Switches to Rust backend for large datasets

### Testing
- Test files in `tests/` directory
- Uses pytest with coverage reporting
- CI/CD runs tests on Python 3.9-3.13
- Configuration in `pyproject.toml` and `tox.ini`

## Important Notes
- Uses `uv` for dependency management and virtual environments
- Fork-based multiprocessing for memory efficiency with Union Find
- Configuration-driven workflow via `config.toml`
- Supports both exact and approximate deduplication modes
- Gradio interface available for result analysis and visualization
