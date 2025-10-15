# =============================================================================
# ML WSL Boilerplate - Makefile for Development Automation
# =============================================================================
# This Makefile provides convenient commands for common development tasks.
# Usage: make <command>
#
# Available commands:
#   setup     - Initial project setup (create environment, install dependencies)
#   test      - Run all tests with coverage
#   test-fast - Run tests without coverage for quick feedback
#   lint      - Run code quality checks (ruff, formatting)
#   format    - Auto-format code with ruff and isort
#   train     - Run model training pipeline
#   notebook  - Start Jupyter Lab server
#   clean     - Clean temporary files and caches
#   env       - Show environment information
#   help      - Show this help message
# =============================================================================

.PHONY: help setup test test-fast lint format train notebook clean env install update

# Default target
.DEFAULT_GOAL := help

# Variables
ENV_NAME = ml-wsl-base
PYTHON = python
PIP = pip
CONDA = conda
MAMBA = mamba

# Detect if mamba is available, use it instead of conda for speed
CONDA_CMD := $(shell command -v mamba 2> /dev/null && echo mamba || echo conda)

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
MAGENTA = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Helper function to print colored output
define print_header
	@echo "$(CYAN)============================================================$(NC)"
	@echo "$(CYAN)  $1$(NC)"
	@echo "$(CYAN)============================================================$(NC)"
endef

help: ## Show this help message
	@echo "$(CYAN)ML WSL Boilerplate - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-12s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make setup     # Initial project setup"
	@echo "  make test      # Run all tests with coverage"
	@echo "  make train     # Train the model"
	@echo "  make notebook  # Start Jupyter Lab"

setup: ## Initial project setup (create environment, install dependencies)
	$(call print_header,SETTING UP ML WSL BOILERPLATE)
	@echo "$(YELLOW)Checking prerequisites...$(NC)"
	@command -v $(CONDA_CMD) >/dev/null 2>&1 || { echo "$(RED)ERROR: $(CONDA_CMD) not found. Please install Conda/Mamba first.$(NC)"; exit 1; }
	@echo "$(GREEN)Using package manager: $(CONDA_CMD)$(NC)"

	@echo "$(YELLOW)Creating/updating environment...$(NC)"
	$(CONDA_CMD) env create -f environment.yml --yes || $(CONDA_CMD) env update -f environment.yml --yes

	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) pre-commit install

	@echo "$(YELLOW)Creating necessary directories...$(NC)"
	@mkdir -p data/raw data/processed data/features models logs outputs/figures outputs/reports

	@echo "$(GREEN)Setup completed! Activate environment with: conda activate $(ENV_NAME)$(NC)"

install: ## Install/update dependencies
	$(call print_header,INSTALLING DEPENDENCIES)
	$(CONDA_CMD) env update -f environment.yml --yes
	@echo "$(GREEN)Dependencies updated$(NC)"

update: install ## Alias for install

env: ## Show environment information
	$(call print_header,ENVIRONMENT INFORMATION)
	@echo "$(YELLOW)Environment Name:$(NC) $(ENV_NAME)"
	@echo "$(YELLOW)Conda/Mamba Command:$(NC) $(CONDA_CMD)"
	@echo "$(YELLOW)Python Version:$(NC)"
	@$(CONDA_CMD) run -n $(ENV_NAME) python --version
	@echo "$(YELLOW)Installed Packages:$(NC)"
	@$(CONDA_CMD) list -n $(ENV_NAME) | head -20
	@echo "..."

test: ## Run all tests with coverage
	$(call print_header,RUNNING TESTS WITH COVERAGE)
	$(CONDA_CMD) run -n $(ENV_NAME) pytest -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Tests completed. Coverage report: htmlcov/index.html$(NC)"

test-fast: ## Run tests without coverage for quick feedback
	$(call print_header,RUNNING FAST TESTS)
	$(CONDA_CMD) run -n $(ENV_NAME) pytest -v -x
	@echo "$(GREEN)Fast tests completed$(NC)"

test-unit: ## Run only unit tests
	$(call print_header,RUNNING UNIT TESTS)
	$(CONDA_CMD) run -n $(ENV_NAME) pytest -v -m "not integration" tests/
	@echo "$(GREEN)Unit tests completed$(NC)"

test-integration: ## Run only integration tests
	$(call print_header,RUNNING INTEGRATION TESTS)
	$(CONDA_CMD) run -n $(ENV_NAME) pytest -v -m "integration" tests/
	@echo "$(GREEN)Integration tests completed$(NC)"

lint: ## Run code quality checks
	$(call print_header,RUNNING CODE QUALITY CHECKS)
	@echo "$(YELLOW)Running Ruff linter...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) ruff check src/ tests/
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) ruff format --check src/ tests/
	@echo "$(YELLOW)Running import sorting check...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) isort --check-only src/ tests/
	@echo "$(GREEN)All quality checks passed$(NC)"

format: ## Auto-format code
	$(call print_header,FORMATTING CODE)
	@echo "$(YELLOW)Running Ruff formatter...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) ruff format src/ tests/
	@echo "$(YELLOW)Sorting imports...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) isort src/ tests/
	@echo "$(YELLOW)Running Ruff auto-fixes...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) ruff check --fix src/ tests/
	@echo "$(GREEN)Code formatting completed$(NC)"

train: ## Run model training pipeline
	$(call print_header,TRAINING MODEL)
	@echo "$(YELLOW)Starting model training...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) python scripts/train.py
	@echo "$(GREEN)Training completed$(NC)"

train-config: ## Train model with custom config file
	$(call print_header,TRAINING WITH CUSTOM CONFIG)
	@read -p "Enter config file path: " config_path; \
	$(CONDA_CMD) run -n $(ENV_NAME) python scripts/train.py --config $$config_path

notebook: ## Start Jupyter Lab server
	$(call print_header,STARTING JUPYTER LAB)
	@echo "$(YELLOW)Starting Jupyter Lab server...$(NC)"
	@echo "$(CYAN)Access at: http://localhost:8888$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) jupyter lab --no-browser --port=8888

notebook-bg: ## Start Jupyter Lab in background
	$(call print_header,STARTING JUPYTER LAB IN BACKGROUND)
	$(CONDA_CMD) run -n $(ENV_NAME) nohup jupyter lab --no-browser --port=8888 > jupyter.log 2>&1 &
	@echo "$(GREEN)Jupyter Lab started in background. Check jupyter.log for details$(NC)"

clean: ## Clean temporary files and caches
	$(call print_header,CLEANING TEMPORARY FILES)
	@echo "$(YELLOW)Removing Python cache files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

	@echo "$(YELLOW)Removing test and coverage files...$(NC)"
	rm -rf .pytest_cache htmlcov .coverage

	@echo "$(YELLOW)Removing temporary files...$(NC)"
	rm -rf .tmp tmp/ *.log

	@echo "$(YELLOW)Removing empty log files...$(NC)"
	find logs/ -name "*.log" -size 0 -delete 2>/dev/null || true

	@echo "$(GREEN)Cleanup completed$(NC)"

clean-data: ## Clean processed data files (keeps raw data)
	$(call print_header,CLEANING PROCESSED DATA)
	@echo "$(YELLOW)Removing processed data files...$(NC)"
	rm -rf data/processed/* data/features/*
	@echo "$(YELLOW)Removing model files...$(NC)"
	rm -rf models/*
	@echo "$(YELLOW)Removing output files...$(NC)"
	rm -rf outputs/*
	@echo "$(GREEN)Data cleanup completed$(NC)"

clean-all: clean clean-data ## Clean everything including data
	$(call print_header,COMPLETE CLEANUP)
	@echo "$(YELLOW)Removing MLflow runs...$(NC)"
	rm -rf mlruns/
	@echo "$(GREEN)Complete cleanup finished$(NC)"

docker-build: ## Build Docker image
	$(call print_header,BUILDING DOCKER IMAGE)
	docker build -t ml-wsl-boilerplate .
	@echo "$(GREEN)Docker image built$(NC)"

docker-run: ## Run Docker container
	$(call print_header,RUNNING DOCKER CONTAINER)
	docker run -it --rm -v $(PWD):/workspace ml-wsl-boilerplate
	@echo "$(GREEN)Docker container started$(NC)"

check: lint test ## Run all quality checks (lint + test)

ci: clean lint test ## Run CI pipeline (clean, lint, test)
	@echo "$(GREEN)CI pipeline completed successfully$(NC)"

dev: format test-fast ## Quick development cycle (format + fast tests)
	@echo "$(GREEN)Development cycle completed$(NC)"

# Development helpers
watch-test: ## Run tests in watch mode (requires entr)
	@echo "$(YELLOW)Watching for file changes... (Ctrl+C to stop)$(NC)"
	@echo "Watching for file changes... (Ctrl+C to stop)"
	find src tests -name "*.py" | entr -c make test-fast

install-dev-tools: ## Install development tools
	$(call print_header,INSTALLING DEVELOPMENT TOOLS)
	@echo "$(YELLOW)Installing development tools...$(NC)"
	$(CONDA_CMD) run -n $(ENV_NAME) pip install entr
	@echo "$(GREEN)Development tools installed$(NC)"

info: ## Show project information
	$(call print_header,PROJECT INFORMATION)
	@echo "$(YELLOW)Project:$(NC) ML WSL Boilerplate"
	@echo "$(YELLOW)Version:$(NC) 0.1.0"
	@echo "$(YELLOW)Python:$(NC) 3.10+"
	@echo "$(YELLOW)Environment:$(NC) $(ENV_NAME)"
	@echo ""
	@echo "$(YELLOW)Key directories:$(NC)"
	@echo "  src/        - Source code"
	@echo "  tests/      - Test files"
	@echo "  config/     - Configuration files"
	@echo "  data/       - Data files"
	@echo "  models/     - Trained models"
	@echo "  notebooks/  - Jupyter notebooks"
	@echo "  logs/       - Log files"

# Quick aliases
t: test-fast ## Alias for test-fast
f: format ## Alias for format
l: lint ## Alias for lint
c: clean ## Alias for clean
