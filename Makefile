.PHONY: test coverage precommit docs test_subset

test:
	coverage run -m pytest
	coverage report -m

test_subset:
	pytest -v test/test_model_bayesian_bivariate.py test/test_model_bayesian_skellam.py test/test_model_bayesian_random_intercept.py test/test_model_bayesian_hierarchical.py

coverage:
	coverage run -m pytest
	coverage report -m

precommit:
	pre-commit install
	pre-commit run --all-files

docs:
	cd docs && make html

jupyter:
	jupyter lab --no-browser --port=8080 --ip="*"
