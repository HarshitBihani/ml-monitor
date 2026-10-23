"""End to end demo: train, deploy, then replay 10 months of production and watch what happens.

    python run_demo.py
Writes registry.db, models/, outputs/results.json (the report is built by report.py from these).
"""
import json
import os
import time
import pandas as pd

from mlmon import features, registry, train, monitor

TRAIN_CUTOFFS = features.month_ends("2010-06-01", "2010-10-31")   # 5 monthly snapshots, labels all known by 31 Jan 2011
PROD_CUTOFFS = features.month_ends("2010-11-01", "2011-08-31")    # 10 months replayed as "production"


def main():
    t0 = time.time()
    registry.reset()
    con = registry.connect()
    sales, returns = features.load()
    print("building snapshots...")
    hist = [features.snapshot(sales, returns, c) for c in TRAIN_CUTOFFS]
    prod = [features.snapshot(sales, returns, c) for c in PROD_CUTOFFS]
    print(f"  train: {sum(len(s) for s in hist):,} rows over {len(hist)} snapshots | production: {sum(len(s) for s in prod):,} rows over {len(prod)} snapshots")

    print("training...")
    info = train.fit_and_register(con, hist)
    print(f"  champion v{info['version']} {info['algo']}  val AUC {info['val_auc']:.3f}  (baseline rule {info['baseline_auc']:.3f})")

    print("replaying production...")
    reference = pd.concat(hist, ignore_index=True)
    timeline = monitor.replay(con, reference, hist, prod, info["val_auc"])
    for row in timeline:
        auc = "   -  " if row["matured_auc"] is None else f"{row['matured_auc']:.3f}"
        print(f"  {row['cutoff']}  v{row['version']}  n={row['customers']:>5}  alert_features={row['alert_features']}  AUC(matured)={auc}  {row['trigger']} {row['outcome']}")

    os.makedirs("outputs", exist_ok=True)
    json.dump(dict(training=info, timeline=timeline, seconds=round(time.time() - t0, 1)), open("outputs/results.json", "w"), indent=2, default=str)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
