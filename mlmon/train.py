"""Training: compare a dumb rule, a simple model and tuned models, then register the best one.

Every candidate is judged on a later period than it was trained on (time-forward validation),
because that is how the model will really be used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES
from . import registry


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Baseline: customers who bought more recently are more likely to buy again. No learning at all."""
    return -df["recency_days"].to_numpy(dtype=float)


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def _make(algo: str, params: dict):
    if algo == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **params))
    return HistGradientBoostingClassifier(random_state=0, **params)


CANDIDATES = [("logistic_regression", {"C": 1.0})] + [
    ("hist_gradient_boosting", dict(learning_rate=lr, max_depth=d, max_iter=200))
    for lr in (0.05, 0.1) for d in (3, 5)
]


def search(con, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Try every candidate, log each as an experiment, return the best by validation AUC."""
    best = None
    for algo, params in CANDIDATES:
        m = _make(algo, params).fit(train_df[FEATURES], train_df["label"])
        auc = _auc(val_df["label"], m.predict_proba(val_df[FEATURES])[:, 1])
        exp = registry.log_experiment(con, "candidate", algo, params, {"val_auc": auc})
        if best is None or auc > best["auc"]:
            best = dict(auc=auc, algo=algo, params=params, model=m, exp=exp)
    return best


def fit_and_register(con, snapshots: list[pd.DataFrame], note: str = "initial training"):
    """snapshots are in time order. The newest one is the validation set, the rest are training data."""
    train_df = pd.concat(snapshots[:-1], ignore_index=True)
    val_df = snapshots[-1]
    base_auc = _auc(val_df["label"], rule_score(val_df))
    registry.log_experiment(con, "baseline_rule", "recency_rule", {}, {"val_auc": base_auc})
    best = search(con, train_df, val_df)
    # refit the winner on everything, so the deployed model has also seen the newest labels
    full = pd.concat(snapshots, ignore_index=True)
    final = _make(best["algo"], best["params"]).fit(full[FEATURES], full["label"])
    trained_on = f"{snapshots[0]['cutoff'].iloc[0]} to {snapshots[-1]['cutoff'].iloc[0]}"
    version = registry.register_model(con, final, best["exp"], best["algo"], best["auc"], trained_on,
                                      stage="champion", note=note)
    return dict(version=version, algo=best["algo"], params=best["params"], val_auc=best["auc"],
                baseline_auc=base_auc, trained_on=trained_on)


def fit_and_register_candidate(con, train_snaps: list[pd.DataFrame], holdout: pd.DataFrame, note: str):
    """Retrain on newer data. The challenger never sees the holdout, so comparing it with the champion is fair."""
    train_df = pd.concat(train_snaps, ignore_index=True)
    best = search(con, train_df, holdout)
    trained_on = f"{train_snaps[0]['cutoff'].iloc[0]} to {train_snaps[-1]['cutoff'].iloc[0]}"
    version = registry.register_model(con, best["model"], best["exp"], best["algo"], best["auc"], trained_on,
                                      stage="challenger", note=note)
    return dict(version=version, auc=best["auc"], algo=best["algo"])
