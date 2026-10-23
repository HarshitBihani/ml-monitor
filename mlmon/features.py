"""Turn raw sales into a prediction problem: "will this customer buy again in the next 90 days?"

For every month-end (a "snapshot") we look back from that date, build a few simple customer features, and
look forward 90 days to see whether the customer really bought. Because it is built from dates, the same
code gives training data from the past and "production" data from later months, with real label delay.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HORIZON = 90          # days we look ahead for the label
WINDOW = 180          # days we look back for activity features
FEATURES = ["recency_days", "frequency_180", "monetary_180", "tenure_days", "orders_total",
            "avg_basket", "distinct_products_180", "return_lines_180", "is_uk"]


def load():
    sales = pd.read_parquet(os.path.join(DATA, "sales.parquet"))
    returns = pd.read_parquet(os.path.join(DATA, "returns.parquet"))
    sales = sales[sales.customer_id.notna()].copy()
    returns = returns[returns.customer_id.notna()].copy()
    sales["customer_id"] = sales.customer_id.astype("int64")
    returns["customer_id"] = returns.customer_id.astype("int64")
    return sales, returns


def snapshot(sales: pd.DataFrame, returns: pd.DataFrame, cutoff: str, with_labels: bool = True) -> pd.DataFrame:
    """Features as they were known at `cutoff` (end of that day). Label uses the 90 days after."""
    t = pd.Timestamp(cutoff) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    past = sales[sales.invoice_date <= t]
    win = past[past.invoice_date > t - pd.Timedelta(days=WINDOW)]
    g_all = past.groupby("customer_id")
    base = pd.DataFrame({
        "recency_days": (t - g_all.invoice_date.max()).dt.days,
        "tenure_days": (t - g_all.invoice_date.min()).dt.days,
        "orders_total": g_all.invoice.nunique(),
        "avg_basket": g_all.revenue.sum() / g_all.invoice.nunique(),
        "is_uk": g_all.country.agg(lambda s: int((s == "United Kingdom").mean() > 0.5)),
    })
    g_w = win.groupby("customer_id")
    w = pd.DataFrame({
        "frequency_180": g_w.invoice.nunique(),
        "monetary_180": g_w.revenue.sum(),
        "distinct_products_180": g_w.stock_code.nunique(),
    })
    r = returns[(returns.invoice_date <= t) & (returns.invoice_date > t - pd.Timedelta(days=WINDOW))]
    rl = r.groupby("customer_id").size().rename("return_lines_180")
    df = base.join(w).join(rl).fillna({"frequency_180": 0, "monetary_180": 0, "distinct_products_180": 0, "return_lines_180": 0})
    df["avg_basket"] = df.avg_basket.round(2)
    df.insert(0, "cutoff", pd.Timestamp(cutoff).date())
    if with_labels:
        fut = sales[(sales.invoice_date > t) & (sales.invoice_date <= t + pd.Timedelta(days=HORIZON))]
        df["label"] = df.index.isin(fut.customer_id.unique()).astype(int)
    return df.reset_index().rename(columns={"index": "customer_id"})


def month_ends(start: str, end: str):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="ME")]
