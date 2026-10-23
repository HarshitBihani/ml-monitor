import os, sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlmon import api, features, registry, train


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("reg")
    registry.DB_PATH = str(d / "r.db")
    registry.MODEL_DIR = str(d / "models")
    api.registry.DB_PATH = registry.DB_PATH
    # connect() takes its default path at definition time, so patch the default
    registry.connect.__defaults__ = (registry.DB_PATH,)
    con = registry.connect()
    sales, returns = features.load()
    snaps = [features.snapshot(sales, returns, c) for c in features.month_ends("2010-06-01", "2010-10-31")]
    train.fit_and_register(con, snaps)
    con.close()
    api._state.update(version=None, model=None)
    return TestClient(api.app), snaps[-1]


def test_health(client):
    c, _ = client
    r = c.get("/health").json()
    assert r["status"] == "ok" and r["model_version"] == 1


def test_predict_returns_valid_probabilities(client):
    c, snap = client
    rows = snap[features.FEATURES].head(20).to_dict("records")
    r = c.post("/predict", json={"cutoff": "t", "rows": rows}).json()
    assert len(r["scores"]) == 20 and all(0 <= s <= 1 for s in r["scores"]) and r["latency_ms"] >= 0


def test_predict_rejects_missing_features(client):
    c, snap = client
    row = snap[features.FEATURES].head(1).to_dict("records")[0]
    row.pop("recency_days")
    assert c.post("/predict", json={"cutoff": "t", "rows": [row]}).status_code == 422


def test_predict_rejects_empty_batch(client):
    c, _ = client
    assert c.post("/predict", json={"cutoff": "t", "rows": []}).status_code == 422
