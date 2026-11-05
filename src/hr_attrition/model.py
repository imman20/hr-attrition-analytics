"""Saya memisahkan kelompok karyawan dan tanggal evaluasi dari pelatihan."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, SEED, snapshot


def make_model(kind):
    pre = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    estimator = (
        LogisticRegression(max_iter=1500, random_state=SEED)
        if kind == "Logistic regression"
        else RandomForestClassifier(n_estimators=250, min_samples_leaf=15, max_depth=7, random_state=SEED, n_jobs=1)
    )
    return Pipeline([("preprocess", pre), ("estimator", estimator)])


def split_cohorts(frame):
    train_ids, test_ids = train_test_split(frame.employee_id, test_size=0.3, random_state=SEED)
    train = snapshot(frame[frame.employee_id.isin(train_ids)], "2023-01-01")
    test = snapshot(frame[frame.employee_id.isin(test_ids)], "2024-01-01")
    return train, test


def evaluate(y, p, threshold):
    p = np.asarray(p, dtype=float)
    return dict(
        auc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        prevalence=float(np.mean(y)),
        confusion_matrix=confusion_matrix(y, p >= threshold, labels=[0, 1]).tolist(),
    )


def fit_models(frame):
    train, test = split_cohorts(frame)
    fit, valid = train_test_split(train, test_size=0.25, stratify=train.target, random_state=SEED)
    models, metrics, validation = {}, {}, {}
    for name in ["Logistic regression", "Random forest"]:
        model = make_model(name).fit(fit[FEATURES], fit.target)
        p = model.predict_proba(valid[FEATURES])[:, 1]
        validation[name] = float(average_precision_score(valid.target, p))
        # Ambang 0,5 ditetapkan sebelumnya; evaluasi memakai confusion matrix yang sama.
        metrics[name] = dict(validation_ap=validation[name])
        models[name] = make_model(name).fit(train[FEATURES], train.target)
        test_p = models[name].predict_proba(test[FEATURES])[:, 1]
        metrics[name].update(evaluate(test.target, test_p, 0.5))
    selected = max(validation, key=validation.get)
    return models, metrics, selected, train, test


def risk_table(model, current, reference):
    """Saya membandingkan probabilitas dengan nilai acuan training per fitur, tanpa klaim sebab."""
    result = current[["employee_id", "department", "job_title"]].copy()
    base = model.predict_proba(current[FEATURES])[:, 1]
    contributions = {}
    for feature in FEATURES:
        alternative = current[FEATURES].copy()
        alternative[feature] = reference[feature].median() if feature in NUMERIC else reference[feature].mode().iloc[0]
        contributions[feature] = base - model.predict_proba(alternative)[:, 1]
    delta = pd.DataFrame(contributions, index=current.index)
    result["risk_12m"] = base
    result["drivers"] = delta.apply(lambda row: ", ".join(f"{k} ({v:+.3f})" for k, v in row.nlargest(3).items()), axis=1)
    return result.sort_values("risk_12m", ascending=False)
