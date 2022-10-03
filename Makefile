.PHONY: test coverage precommit docs test_subset

test:
	poetry run coverage run -m pytest

test_subset:
	poetry run pytest -v test/test_model_bayesian_hierarchical.py test/test_model_bayesian_bivariate.py test/test_model_bayesian_random_intercept.py

coverage:
	poetry run coverage run -m pytest
	poetry run coverage report -m

precommit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

docs:
	cd docs && poetry run make html
