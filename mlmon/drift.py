"""Data drift measures.

PSI (Population Stability Index): how much a feature's distribution moved away from the training data.
Common rule of thumb: below 0.1 stable, 0.1 to 0.2 some shift, above 0.2 a big shift.
KS statistic: the largest gap between the two cumulative distributions (0 = identical, 1 = completely different).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp

PSI_WARN, PSI_ALERT = 0.1, 0.2


def psi(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    # bin edges come from the reference data, so "the same bins" mean the same thing every month
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # almost constant feature (for example a 0/1 flag)
        edges = np.array([-np.inf, ref.mean(), np.inf]) if ref.std() > 0 else np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference_df, current_df, features):
    out = {}
    for f in features:
        out[f] = dict(psi=psi(reference_df[f], current_df[f]),
                      ks=float(ks_2samp(reference_df[f], current_df[f]).statistic))
    return out


def level(p: float) -> str:
    return "alert" if p >= PSI_ALERT else "warn" if p >= PSI_WARN else "ok"
