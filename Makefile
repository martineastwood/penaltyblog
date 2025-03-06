.PHONY: test coverage precommit docs test_subset

test:
	coverage run -m pytest
	coverage report -m

test_subset:
	pytest -v test/test_metrics_rps.py

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

golib:
	cd penaltyblog/gosrc && go build -o ../golib/penaltyblog.dylib -buildmode=c-shared .
