PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: install research test smoke synthetic benchmark validate clean

install:
	$(PYTHON) -m pip install -e .

research:
	$(PYTHON) -m pip install -e '.[research]'

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/run_smoke.py

synthetic:
	$(PYTHON) scripts/generate_synthetic.py --all

benchmark:
	$(PYTHON) scripts/run_benchmark.py --config configs/benchmark.yaml

validate:
	$(PYTHON) scripts/validate_repository.py

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
