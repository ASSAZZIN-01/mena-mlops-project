.PHONY: test lint check

test:
	python -m pytest

lint:
	python -m ruff check .

check: lint test
