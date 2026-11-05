"""Saya merekonsiliasi artefak yang diterbitkan dengan prediksi holdout."""

import json

import nbformat
import pandas as pd
import pytest

from hr_attrition.data import ROOT
from hr_attrition.model import evaluate


def test_published_metrics_match_prediction_rows():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "reports/test_predictions.csv")
    assert len(predictions) == metrics["test_n"]
    for name, expected in metrics["models"].items():
        actual = evaluate(predictions.target, predictions[name], 0.5)
        for key in ["auc", "average_precision", "brier", "prevalence"]:
            assert actual[key] == pytest.approx(expected[key], abs=1e-12)
        assert actual["confusion_matrix"] == expected["confusion_matrix"]


def test_notebook_has_executed_cells_without_errors():
    notebook = nbformat.read(ROOT / "notebooks/01_hr_attrition.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) >= 5
    assert all(cell.execution_count is not None for cell in code)
    assert not any(out.output_type == "error" for cell in code for out in cell.outputs)
