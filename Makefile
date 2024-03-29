.PHONY: test coverage precommit docs test_subset

test:
	poetry run coverage run -m pytest

test_subset:
	poetry run pytest -v test/test_backtest.py

coverage:
	poetry run coverage run -m pytest
	poetry run coverage report -m

precommit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

docs:
	cd docs && poetry run make html

jupyter:
	poetry run jupyter lab --no-browser --port=8080 --ip="*"
