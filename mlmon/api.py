"""Prediction API. Loads the current champion model from the registry and logs every batch it scores.

  uvicorn mlmon.api:app --port 8000
  POST /predict   {"cutoff": "2011-03-31", "rows": [{...features...}, ...]}  ->  scores
  POST /reload    pick up a newly promoted champion without restarting
  GET  /health    which model version is live
"""
from __future__ import annotations
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import registry
from .features import FEATURES

app = FastAPI(title="Repeat-purchase model API")
_state = {"version": None, "model": None}


def _load():
    con = registry.connect()
    champ = registry.champion(con)
    con.close()
    if champ is None:
        raise HTTPException(503, "No champion model registered yet. Run training first.")
    _state["version"], _state["model"] = champ["version"], joblib.load(champ["path"])


class Batch(BaseModel):
    cutoff: str = Field(default="live", description="Snapshot date these rows describe")
    rows: list[dict]


@app.get("/health")
def health():
    if _state["model"] is None:
        _load()
    return {"status": "ok", "model_version": _state["version"]}


@app.post("/reload")
def reload():
    _load()
    return {"model_version": _state["version"]}


@app.post("/predict")
def predict(batch: Batch):
    if _state["model"] is None:
        _load()
    if not batch.rows:
        raise HTTPException(422, "rows is empty")
    df = pd.DataFrame(batch.rows)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    t0 = time.perf_counter()
    scores = _state["model"].predict_proba(df[FEATURES])[:, 1]
    ms = (time.perf_counter() - t0) * 1000
    con = registry.connect()
    con.execute("INSERT INTO predictions (ts,version,cutoff,n,mean_score,latency_ms) VALUES (datetime('now'),?,?,?,?,?)",
                (_state["version"], batch.cutoff, len(df), float(scores.mean()), ms))
    con.commit()
    con.close()
    return {"model_version": _state["version"], "scores": [round(float(s), 5) for s in scores],
            "latency_ms": round(ms, 2)}
