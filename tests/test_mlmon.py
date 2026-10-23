import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlmon import drift, features, registry, train


# ---------- drift ----------
def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert drift.psi(a, b) < 0.05


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    small, big = drift.psi(a, rng.normal(0.3, 1, 5000)), drift.psi(a, rng.normal(1.5, 1, 5000))
    assert small < big and big > drift.PSI_ALERT


def test_psi_handles_constant_and_binary_features():
    ref = np.array([0] * 80 + [1] * 20)
    assert drift.psi(ref, ref) == pytest.approx(0, abs=1e-6)
    assert drift.psi(np.ones(100), np.ones(100)) == pytest.approx(0, abs=1e-6)


# ---------- features: no peeking into the future ----------
@pytest.fixture(scope="module")
def data():
    return features.load()


def test_features_ignore_future_sales(data):
    sales, returns = data
    cutoff = "2010-09-30"
    a = features.snapshot(sales, returns, cutoff, with_labels=False)
    future = sales[sales.invoice_date > pd.Timestamp(cutoff) + pd.Timedelta(days=1)]
    sales2 = sales.copy()
    sales2.loc[future.index, "revenue"] = 0.0     # destroy all future information
    b = features.snapshot(sales2, returns, cutoff, with_labels=False)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_label_uses_only_the_next_90_days(data):
    sales, returns = data
    snap = features.snapshot(sales, returns, "2010-09-30")
    assert 0.2 < snap.label.mean() < 0.9      # a sane base rate, not all zeros or all ones
    assert snap.customer_id.is_unique


# ---------- registry ----------
def test_registry_champion_and_promotion(tmp_path):
    con = registry.connect(str(tmp_path / "r.db"))
    registry.MODEL_DIR = str(tmp_path / "m")
    e = registry.log_experiment(con, "t", "x", {}, {"val_auc": 0.7})
    v1 = registry.register_model(con, {"a": 1}, e, "x", 0.7, "p1")
    v2 = registry.register_model(con, {"a": 2}, e, "x", 0.8, "p2", stage="challenger")
    assert registry.champion(con)["version"] == v1
    registry.promote(con, v2)
    assert registry.champion(con)["version"] == v2
    stages = [r["stage"] for r in con.execute("select stage from models order by version")]
    assert stages == ["archived", "champion"]


# ---------- training ----------
def test_trained_model_beats_the_simple_rule(data, tmp_path):
    sales, returns = data
    snaps = [features.snapshot(sales, returns, c) for c in features.month_ends("2010-06-01", "2010-10-31")]
    con = registry.connect(str(tmp_path / "r.db"))
    registry.MODEL_DIR = str(tmp_path / "m")
    info = train.fit_and_register(con, snaps)
    assert info["val_auc"] > info["baseline_auc"]
