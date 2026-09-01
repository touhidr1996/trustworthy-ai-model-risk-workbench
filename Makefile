.PHONY: setup run test api
setup:
	python -m pip install -e ".[dev]"
run:
	PYTHONPATH=src python -m trustbench.pipeline
test:
	PYTHONPATH=src pytest -q
api:
	uvicorn trustbench.api:app --app-dir src --reload
