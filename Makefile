.PHONY: test lint check

test:
	uv run pytest

lint:
	uv run ruff check .

check: lint test
