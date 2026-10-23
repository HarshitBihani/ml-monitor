"""Build outputs/report.html from registry.db and outputs/results.json."""
import json
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

con = sqlite3.connect("registry.db")
q = lambda s: pd.read_sql_query(s, con)
res = json.load(open("outputs/results.json"))
perf = q("select * from performance order by cutoff")
drift = q("select * from drift")
models = q("select version, ts, algo, stage, round(val_auc,4) val_auc, trained_on, note from models")
events = q("select cutoff, kind, detail from events order by cutoff")
preds = q("select version, n, latency_ms from predictions where cutoff != 'holdout-check'")
exps = q("select id, algo, params, metrics from experiments where name='candidate' order by id limit 5")
train = res["training"]

lat = preds.latency_ms
per_row_us = (preds.latency_ms / preds.n * 1000)
stats = dict(batches=len(preds), p50=lat.median(), p95=lat.quantile(.95), per_row_us=per_row_us.median())

f1 = go.Figure()
f1.add_scatter(x=perf.cutoff, y=perf.auc, mode="lines+markers", name="Deployed model", line=dict(color="#2563eb"))
f1.add_scatter(x=perf.cutoff, y=perf.baseline_auc, mode="lines+markers", name="Simple rule (recency only)", line=dict(color="#9ca3af", dash="dash"))
f1.add_hline(y=train["val_auc"] - 0.03, line_dash="dot", line_color="#dc2626", annotation_text="retrain trigger (0.03 below validation AUC)")
f1.update_layout(template="plotly_white", title="Real accuracy (AUC) once the 90-day outcome is known", height=380,
                 yaxis_title="AUC", xaxis_title="Snapshot date the prediction was made", legend=dict(orientation="h", y=-0.25))

pv = drift.pivot_table(index="feature", columns="cutoff", values="psi", aggfunc="mean")
f2 = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale="YlOrRd", zmin=0, zmax=0.6,
                          colorbar=dict(title="PSI"), hovertemplate="%{y}<br>%{x}<br>PSI %{z:.2f}<extra></extra>"))
f2.update_layout(template="plotly_white", title="Feature drift vs training data (PSI: above 0.2 is a big shift)", height=380)


def table(df):
    return df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:.3f}")


tl = pd.DataFrame(res["timeline"])
tl["matured_auc"] = tl.matured_auc.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ML Monitor report</title>
<script>{get_plotlyjs()}</script>
<style>body{{font:15px/1.55 Segoe UI,Arial,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;color:#1f2937}}
h1{{font-size:25px}}h2{{margin-top:30px}}table.t{{border-collapse:collapse;width:100%;font-size:13px}}
table.t td,table.t th{{padding:5px 8px;border-bottom:1px solid #e5e7eb;text-align:left}}.k{{display:flex;gap:12px;flex-wrap:wrap}}
.k div{{border:1px solid #e5e7eb;border-radius:10px;padding:10px 14px}}.k b{{display:block;font-size:20px;color:#2563eb}}</style></head><body>
<h1>Model monitoring report: will this customer buy again in 90 days?</h1>
<p>A model was trained on 2010 retail data, deployed behind an API, then ten months of 2011 were replayed as live traffic.
This report shows what the monitoring saw and what the retraining rule did.</p>
<div class="k"><div><b>{train['val_auc']:.3f}</b>model AUC at deployment ({train['algo'].replace('_',' ')})</div>
<div><b>{train['baseline_auc']:.3f}</b>simple recency rule, same data</div>
<div><b>{stats['p50']:.1f} ms</b>median time to score a month of customers (p95 {stats['p95']:.1f} ms)</div>
<div><b>{len(models)}</b>model versions registered</div></div>
<h2>Did accuracy drop?</h2>{f1.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False))}
<h2>Did the data change?</h2>{f2.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False))}
<p><b>What this shows:</b> several features are flagged as drifted from April onward, yet accuracy stayed in a narrow band.
Part of that drift is natural: customer history keeps growing, so features like tenure and total orders drift upward with no change in behaviour.
Drift alerts tell you to look, not that the model is broken. This is why the platform only replaces the model when a retrained
version also wins on fresh labelled data.</p>
<h2>Month by month</h2>{table(tl)}
<h2>Decisions the platform made</h2>{table(events) if len(events) else '<p>No retraining was triggered.</p>'}
<h2>Model registry</h2>{table(models)}
<p>Latency is measured in-process (API code path, JSON handling and scoring, no network). Real network time would add to it.</p>
<p>AUC is measured here on customers who were active before each snapshot date. It measures how well customers are ranked,
not how many are right, so it can be high even when the model needs recalibrating.</p>
</body></html>"""
open("outputs/report.html", "w", encoding="utf-8").write(html)
json.dump({k: (round(float(v), 3)) for k, v in stats.items()}, open("outputs/latency.json", "w"))
print("wrote outputs/report.html", {k: round(float(v), 2) for k, v in stats.items()})
