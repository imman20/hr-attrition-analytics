PYTHON ?= python
export PYTHONPATH := src
.PHONY: all install lint test pipeline app
all: lint test pipeline
install:
	$(PYTHON) -m pip install -r requirements.txt
lint:
	$(PYTHON) -m ruff check src tests app

test:
	$(PYTHON) -m pytest
pipeline:
	$(PYTHON) -m hr_attrition.pipeline
app:
	$(PYTHON) -m streamlit run app/dashboard.py
