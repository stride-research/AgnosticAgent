.PHONY: install lint format test check setup-pre-commit

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .[dev]

setup-pre-commit:
	. .venv/bin/activate && pre-commit install

lint:
	. .venv/bin/activate && ruff check . --fix

format:
	. .venv/bin/activate && black .

test:
	. .venv/bin/activate && tox

check:
	. .venv/bin/activate && tox -e lint,format
