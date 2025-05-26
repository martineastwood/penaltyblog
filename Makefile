.PHONY: test coverage precommit docs test_subset

test:
	coverage run -m pytest
	coverage report -m

test_subset:
	pytest -v test/test_flow.py test/test_flowgroup.py test/test_flow_glob.py test/test_flow_helpers.py test/test_flow_optimizer.py

test_implied:
	pytest -v test/test_implied.py

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

cython:
	python setup.py build_ext --inplace
