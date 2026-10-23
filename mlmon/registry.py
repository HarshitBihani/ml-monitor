"""A small experiment tracker and model registry on SQLite.

Chosen over a heavier tool on purpose: everything is one file you can open and query, and each table is easy to explain.
"""
from __future__ import annotations
import json
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "registry.db")
MODEL_DIR = os.path.join(ROOT, "models")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, ts TEXT, name TEXT, algo TEXT, params TEXT, metrics TEXT);
CREATE TABLE IF NOT EXISTS models (version INTEGER PRIMARY KEY, ts TEXT, experiment_id INTEGER, algo TEXT,
    stage TEXT, path TEXT, val_auc REAL, trained_on TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, ts TEXT, version INTEGER, cutoff TEXT, n INTEGER,
    mean_score REAL, latency_ms REAL);
CREATE TABLE IF NOT EXISTS drift (cutoff TEXT, version INTEGER, feature TEXT, psi REAL, ks REAL);
CREATE TABLE IF NOT EXISTS performance (cutoff TEXT, version INTEGER, auc REAL, baseline_auc REAL, base_rate REAL, n INTEGER);
CREATE TABLE IF NOT EXISTS events (ts TEXT, cutoff TEXT, kind TEXT, detail TEXT);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def reset(path: str = DB_PATH):
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for f in os.listdir(MODEL_DIR):
        os.remove(os.path.join(MODEL_DIR, f))


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_experiment(con, name, algo, params, metrics) -> int:
    cur = con.execute("INSERT INTO experiments (ts,name,algo,params,metrics) VALUES (?,?,?,?,?)",
                      (_now(), name, algo, json.dumps(params, default=str), json.dumps(metrics)))
    con.commit()
    return cur.lastrowid


def register_model(con, model, experiment_id, algo, val_auc, trained_on, stage="champion", note="") -> int:
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    row = con.execute("SELECT COALESCE(MAX(version),0)+1 FROM models").fetchone()[0]
    path = os.path.join(MODEL_DIR, f"model_v{row}.joblib")
    joblib.dump(model, path)
    if stage == "champion":
        con.execute("UPDATE models SET stage='archived' WHERE stage='champion'")
    con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)",
                (row, _now(), experiment_id, algo, stage, path, val_auc, trained_on, note))
    con.commit()
    return row


def promote(con, version: int):
    con.execute("UPDATE models SET stage='archived' WHERE stage='champion'")
    con.execute("UPDATE models SET stage='champion' WHERE version=?", (version,))
    con.commit()


def champion(con):
    return con.execute("SELECT * FROM models WHERE stage='champion' ORDER BY version DESC LIMIT 1").fetchone()


def log_event(con, cutoff, kind, detail):
    con.execute("INSERT INTO events VALUES (?,?,?,?)", (_now(), cutoff, kind, detail))
    con.commit()
