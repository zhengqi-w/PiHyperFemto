PYTHON ?= python3
CONFIG ?= configs/testset.yaml
export PYTHONPATH := src
export OMP_NUM_THREADS := 1

.PHONY: test inspect run validate
test:
	$(PYTHON) -m unittest discover -s tests -v

inspect:
	$(PYTHON) -m pih3l inspect -c $(CONFIG)

run:
	$(PYTHON) -m pih3l run -c $(CONFIG)

validate:
	$(PYTHON) scripts/validate_testset.py
