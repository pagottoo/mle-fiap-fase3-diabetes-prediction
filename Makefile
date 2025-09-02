# Sistema de Predição de Diabetes - Makefile
# Comandos de automação para desenvolvimento

# Variables
PYTHON := python3.11
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
REQUIREMENTS := requirements.txt

# Default target
.PHONY: help
help:
	@echo "Sistema de Predição de Diabetes - Comandos Disponíveis:"
	@echo ""
	@echo "Development:"
	@echo "  make venv        - Create virtual environment" 
	@echo "  make install     - Install dependencies"
	@echo "  make format      - Format code with black and isort"
	@echo "  make lint        - Lint code"
	@echo "  make test        - Run all tests"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make train       - Train ML models"
	@echo ""
	@echo "Services (Local):"
	@echo "  make run-api     - Start FastAPI server"
	@echo "  make run-app     - Start Streamlit dashboard"
	@echo "  make test-api    - Test API endpoints"
	@echo ""
	@echo "Docker Compose:"
	@echo "  make compose-up    - Start all services (db/api/app)"
	@echo "  make compose-down  - Stop all services"
	@echo "  make compose-logs  - Show service logs"
	@echo "  make compose-ps    - Show running services"
	@echo "  make compose-clean - Clean Docker resources"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean temp files"
	@echo "  make clean-all   - Clean everything including venv"
	@echo ""

# Environment setup
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)/"

.PHONY: install
install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@echo "Dependencies installed."

# ML Pipeline
.PHONY: train
train:
	@echo "Training ML models..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON_VENV) -m ml.train
	@echo "Training completed."

# Testing
.PHONY: test
test:
	@echo "Running all tests..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON_VENV) -m pytest tests/ -v

.PHONY: test-api
test-api:
	@echo "Testing API endpoints (requires running API server)..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	@echo "Use 'curl http://localhost:8000/docs' ou acesse http://localhost:8000/docs para testar os endpoints"

# Local services
.PHONY: run-api
run-api:
	@echo "Starting FastAPI server..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON_VENV) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: run-app
run-app:
	@echo "Starting Streamlit dashboard..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON_VENV) -m streamlit run app/main.py --server.port 8501

# Code quality
.PHONY: format
format:
	@echo "Formatting code..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON_VENV) -m black api/ ml/ app/ tests/
	$(PYTHON_VENV) -m isort api/ ml/ app/ tests/

.PHONY: lint
lint:
	@echo "Linting code..."
	@if [ ! -d "$(VENV)" ]; then echo "Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON_VENV) -m flake8 api/ ml/ app/ tests/ --max-line-length=100

# Docker Compose commands
.PHONY: compose-up
compose-up:
	@echo "Starting all services with Docker Compose..."
	docker compose up --build -d
	@echo "Services started. Access:"
	@echo "  - API: http://localhost:8000"
	@echo "  - App: http://localhost:8501"
	@echo "  - Database: postgresql://postgres:postgres@localhost:5432/diabetes_health"

.PHONY: compose-down
compose-down:
	@echo "Stopping all services..."
	docker compose down
	@echo "All services stopped."

.PHONY: compose-logs
compose-logs:
	@echo "Showing service logs..."
	docker compose logs -f

.PHONY: compose-ps
compose-ps:
	@echo "Showing running services..."
	docker compose ps

.PHONY: compose-clean
compose-clean:
	@echo "Cleaning Docker Compose resources..."
	docker compose down -v --rmi all
	docker system prune -f
	@echo "Cleanup completed."

# Utility commands
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	@echo "Cleanup completed."

.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Full cleanup completed."

# Database commands (placeholder for future use)
.PHONY: db-init
db-init:
	@echo "Initializing database..."
	@echo "Database initialization completed."

.PHONY: db-migrate
db-migrate:
	@echo "Running database migrations..."
	@echo "Database migrations completed."