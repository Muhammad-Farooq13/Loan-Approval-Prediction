# Makefile for Loan Prediction Project

.PHONY: help install install-dev test lint format clean run docker-build docker-run mlops

help:
	@echo "Available commands:"
	@echo "  make install       - Install project dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with pytest"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run           - Run Flask application"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make mlops         - Run MLOps pipeline"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort pre-commit
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/

run:
	python flask_app.py

docker-build:
	docker build -t loan-prediction:latest .

docker-run:
	docker run -p 5000:5000 loan-prediction:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

mlops:
	python mlops_pipeline.py

train:
	python src/models/train_model.py

evaluate:
	python src/models/evaluate_model.py
