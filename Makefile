.PHONY: help install install-dev test lint format clean build docs

help: ## Show this help message
	@echo "CircuitKit Development Commands"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install CircuitKit in development mode
	pip install -e .

install-dev: ## Install CircuitKit with development dependencies
	pip install -e .[dev]
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src/circuitkit --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

lint: ## Run linting checks
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	mypy src/ --ignore-missing-imports

format: ## Format code with black and isort
	black src/ tests/ examples/
	isort src/ tests/ examples/ --profile=black

format-check: ## Check code formatting
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/ --profile=black

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

cli-test: ## Test CLI commands
	circuitkit --help
	circuitkit discover --help
	circuitkit evaluate --help
	circuitkit list-models

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

run-pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

security-check: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json
	safety check

ci: ## Run CI checks locally
	make format-check
	make lint
	make test
	make security-check

all: clean install-dev test lint format ## Run all checks and tests
