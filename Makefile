PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: setup check lint typecheck test train-fast export backend frontend up

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-train.txt
	$(PIP) install -r backend/requirements.txt

check: lint typecheck test

lint:
	ruff check .

typecheck:
	mypy common training tremor backend tests

test:
	pytest -q

train-fast:
	$(PYTHON) training/train.py --fast --config configs/train_resnet50.yaml

export:
	$(PYTHON) training/export_onnx.py --config configs/export.yaml

backend:
	cd backend && uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

up:
	docker compose up --build
