# Text Dedup Agent Configuration

## Commands
- **Test**: `just test` (all tests), `uv run pytest tests/test_specific.py::test_name` (single test)
- **Type check**: `just check` (full checks), `uv run mypy` (type checking only)
- **Lint**: `uv run ruff check .` (linting), `uv run ruff format .` (formatting)
- **Build**: `just build` (wheel), `just install` (dev environment)
- **Run**: `just run` (with config.toml), `uv run -m text_dedup.minhash`

## Architecture
- Python package for text deduplication with MinHash LSH algorithm
- Main module: `src/text_dedup/` with config, data sources, utilities
- Extension: Rust module in `extensions/rust/` workspace member
- Config-driven execution via `config.toml` with HuggingFace datasets support

## Code Style
- Line length: 120 chars, Python 3.9+ target, single-line imports (ruff isort)
- Type hints required (`disallow_untyped_defs = true`)
- Use `from typing_extensions` for newer type features
- Error handling: prefer explicit exceptions, no lambda assignments (E731)
- Test files exempt from bandit security checks (S101)
