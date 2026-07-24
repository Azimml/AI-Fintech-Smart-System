.PHONY: help install install-dev nltk-data test lint format check run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	python -m pip install -r requirements.txt

install-dev:  ## Install development dependencies (pytest, ruff)
	python -m pip install -r requirements-dev.txt

nltk-data:  ## Download the NLTK corpora used for preprocessing
	python -c "from finbot.preprocessing import ensure_nltk_data; ensure_nltk_data()"

test:  ## Run the test suite
	python -m pytest

lint:  ## Run ruff lint checks
	ruff check src tests

format:  ## Auto-fix lint issues and format with ruff
	ruff check --fix src tests
	ruff format src tests

check: lint test  ## Run lint and tests

run:  ## Classify a single message: make run MSG="What is my balance?"
	python -m finbot "$(MSG)"

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
