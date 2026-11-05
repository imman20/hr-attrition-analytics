"""Saya menguji batas tanggal, label, dan penyebut metrik HR."""

import numpy as np
import pandas as pd
import pytest

from hr_attrition.data import END, FEATURES, PROTECTED, generate, snapshot, turnover
from hr_attrition.model import split_cohorts


def test_generator_is_deterministic_and_temporally_valid():
    frame = generate()
    pd.testing.assert_frame_equal(frame, generate())
    assert len(frame) == frame.employee_id.nunique() == 1500
    assert frame.hire_date.between("2022-01-01", "2022-12-31").all()
    assert (frame.exit_date.dropna() > frame.loc[frame.exit_date.notna(), "hire_date"]).all()
    assert (frame.exit_date.dropna() < END).all()
    assert frame.engagement.between(1, 5).all()
    assert (frame.monthly_salary > 0).all()


def test_snapshot_tenure_uses_prediction_date_not_final_record():
    frame = generate()
    snap = snapshot(frame, "2023-01-01")
    expected = (pd.Timestamp("2023-01-01") - snap.hire_date).dt.days / 365.25
    np.testing.assert_allclose(snap.tenure_years, expected)
    assert set(PROTECTED).isdisjoint(FEATURES)
    assert not set(["exit_date", "exit_reason", "employee_id"]).intersection(FEATURES)


def test_future_promotion_cannot_change_past_snapshot():
    frame = generate(50)
    a = snapshot(frame, "2023-01-01")
    frame["last_promotion_date"] = pd.Timestamp("2026-01-01")
    b = snapshot(frame, "2023-01-01")
    no_past = a.last_promotion_date.isna() | (a.last_promotion_date >= pd.Timestamp("2023-01-01"))
    np.testing.assert_allclose(a.loc[no_past, "months_since_promotion"], b.loc[no_past, "months_since_promotion"])


def test_labels_require_full_followup():
    with pytest.raises(ValueError):
        snapshot(generate(10), "2025-01-01")
    assert len(snapshot(generate(10), "2025-01-01", labels=False)) <= 10


def test_model_groups_do_not_overlap():
    train, test = split_cohorts(generate())
    assert set(train.employee_id).isdisjoint(test.employee_id)
    assert train.tenure_years.max() <= 1.01
    assert test.tenure_years.min() >= 1
    assert train.target.isin([0, 1]).all()


def test_daily_headcount_denominator_and_exit_boundary():
    data = pd.DataFrame(
        dict(
            hire_date=pd.to_datetime(["2022-01-01", "2022-01-01"]),
            exit_date=pd.to_datetime(["2022-07-02", None]),
            department=["SDM", "SDM"],
            exit_reason=["Sukarela", "Aktif"],
            tenure_years=[182 / 365.25, 3],
        )
    )
    monthly, annual = turnover(data)
    row = annual[(annual.year == 2022) & (annual.department == "Semua")].iloc[0]
    assert row.average_headcount == pytest.approx((182 + 365) / 365)
    assert row.turnover_rate == pytest.approx(365 / 547)
    assert row.voluntary_rate + row.involuntary_rate == row.turnover_rate
    assert monthly.query("month == '2022-07' and department == 'SDM'").headcount.iloc[0] == 1


def test_exit_on_landmark_is_event_in_next_year():
    frame = generate(5)
    frame.loc[0, "exit_date"] = pd.Timestamp("2023-01-01")
    row = snapshot(frame, "2023-01-01").set_index("employee_id").loc["E0001"]
    assert row.target == 1
