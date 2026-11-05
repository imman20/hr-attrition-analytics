"""Saya memeriksa evaluasi probabilitas dan kontrak fitur model."""

import numpy as np
import pandas as pd
import pytest

from hr_attrition.data import FEATURES, generate
from hr_attrition.model import evaluate, make_model, risk_table, split_cohorts


def test_evaluate_uses_fixed_class_order():
    result = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9], 0.5)
    assert result["auc"] == 1
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["brier"] == pytest.approx(0.055)


@pytest.mark.parametrize("kind", ["Logistic regression", "Random forest"])
def test_prediction_ignores_outcome_and_protected_columns(kind):
    train, test = split_cohorts(generate(400))
    model = make_model(kind).fit(train[FEATURES], train.target)
    p = model.predict_proba(test[FEATURES])[:, 1]
    mutated = test.copy()
    mutated["exit_date"] = pd.NaT
    mutated["gender"] = "Tidak diketahui"
    mutated["age"] = 99
    mutated["tenure_years"] = test.tenure_years
    np.testing.assert_array_equal(p, model.predict_proba(mutated[FEATURES])[:, 1])
    assert ((p >= 0) & (p <= 1)).all()
    risk = risk_table(model, test.head(8), train)
    assert risk.risk_12m.is_monotonic_decreasing
    assert risk.drivers.str.len().gt(0).all()
