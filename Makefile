.PHONY: test coverage precommit docs test_subset

test:
	coverage run -m pytest
	coverage report -m

test_subset:
	pytest -v test/test_kelly_criterion.py

coverage:
	coverage run -m -F pytest
	coverage report -m

precommit:
	pre-commit install
	pre-commit run --all-files

docs:
	cd docs && make html

jupyter:
	jupyter lab --no-browser --port=8080 --ip="*"
