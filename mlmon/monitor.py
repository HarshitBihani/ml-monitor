"""Replay "production" month by month and watch the model.

At each month-end:
  1. build that month's customer snapshot and score it through the API
  2. measure feature drift against the training data right away (no labels needed)
  3. measure real accuracy for older snapshots whose 90-day outcome is now known (labels arrive late)
  4. apply the retraining rule, and only promote a new model if it beats the current one on fresh data

Retraining rule (written down so it is a decision, not a mood):
  retrain when  AUC on the newest matured snapshot is more than 0.03 below the AUC measured when the champion was chosen
          or    3 or more features show alert-level drift (PSI >= 0.2)
  promote the retrained model only if its AUC on a held-out recent snapshot beats the champion by at least 0.005.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fastapi.testclient import TestClient

from . import api, drift, registry, train
from .features import FEATURES, HORIZON

AUC_DROP_TRIGGER = 0.03
ALERT_FEATURES_TRIGGER = 3
PROMOTE_MARGIN = 0.005
COOLDOWN_MONTHS = 3        # do not retrain again right after a retrain, labels need time to arrive


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def replay(con, reference: pd.DataFrame, history: list[pd.DataFrame], prod_snapshots: list[pd.DataFrame],
           deployed_val_auc: float):
    """history = training snapshots (reused when retraining); prod_snapshots = production months in time order."""
    api._state.update(version=None, model=None)
    client = TestClient(api.app)
    client.post("/reload")
    seen: list[pd.DataFrame] = []
    timeline = []
    reference_auc = deployed_val_auc
    last_retrain_idx = -99

    for i, snap in enumerate(prod_snapshots):
        cutoff = str(snap["cutoff"].iloc[0])
        now = pd.Timestamp(cutoff)
        version = client.get("/health").json()["model_version"]

        # 1. score through the API
        resp = client.post("/predict", json={"cutoff": cutoff, "rows": snap[FEATURES].to_dict("records")}).json()
        snap = snap.assign(score=resp["scores"], model_version=version)
        seen.append(snap)

        # 2. drift against the training data
        d = drift.feature_drift(reference, snap, FEATURES)
        for f, v in d.items():
            con.execute("INSERT INTO drift VALUES (?,?,?,?,?)", (cutoff, version, f, v["psi"], v["ks"]))
        n_alert = sum(v["psi"] >= drift.PSI_ALERT for v in d.values())
        con.commit()          # release the write lock: the API opens its own connection to the same file

        # 3. real accuracy for the newest snapshot whose outcome window has closed
        matured = [s for s in seen if pd.Timestamp(str(s["cutoff"].iloc[0])) + pd.Timedelta(days=HORIZON) <= now]
        newest_auc = None
        if matured:
            m = matured[-1]
            newest_auc = _auc(m["label"], m["score"])
            con.execute("INSERT INTO performance VALUES (?,?,?,?,?,?)",
                        (str(m["cutoff"].iloc[0]), int(m["model_version"].iloc[0]), newest_auc,
                         _auc(m["label"], train.rule_score(m)), float(m["label"].mean()), len(m)))
            con.commit()

        # 4. retraining rule
        reason, outcome = None, ""
        if newest_auc is not None and newest_auc < reference_auc - AUC_DROP_TRIGGER:
            reason = f"AUC {newest_auc:.3f} is more than {AUC_DROP_TRIGGER} below reference {reference_auc:.3f}"
        elif n_alert >= ALERT_FEATURES_TRIGGER:
            reason = f"{n_alert} features at alert-level drift"
        if reason and matured and i - last_retrain_idx >= COOLDOWN_MONTHS:
            last_retrain_idx = i
            registry.log_event(con, cutoff, "retrain_triggered", reason)
            snaps = history + [s.drop(columns=["score", "model_version"]) for s in matured]
            holdout = snaps[-1]
            cand = train.fit_and_register_candidate(con, snaps[:-1], holdout, note=f"retrain at {cutoff}: {reason}")
            champ_auc = _auc(holdout["label"], api._state["model"].predict_proba(holdout[FEATURES])[:, 1])
            if cand["auc"] >= champ_auc + PROMOTE_MARGIN:
                registry.promote(con, cand["version"])
                client.post("/reload")
                reference_auc = cand["auc"]
                outcome = f"promoted v{cand['version']} (AUC {cand['auc']:.3f} vs champion {champ_auc:.3f})"
                registry.log_event(con, cutoff, "promoted", outcome)
            else:
                outcome = f"kept champion (challenger {cand['auc']:.3f} vs champion {champ_auc:.3f})"
                registry.log_event(con, cutoff, "kept_champion", outcome)

        timeline.append(dict(cutoff=cutoff, version=version, customers=len(snap), alert_features=n_alert,
                             matured_auc=newest_auc, trigger=reason or "", outcome=outcome,
                             latency_ms=resp["latency_ms"]))
    return timeline
