.PHONY: test coverage precommit

test:
	poetry run coverage run -m pytest

coverage:
	poetry run coverage run -m pytest
	poetry run coverage report -m

precommit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files
