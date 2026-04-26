.PHONY: setup test train-smoke

setup:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

test:
	python -m pytest

train-smoke:
	python -m pytest tests/test_train_smoke.py -q
