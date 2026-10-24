"""ALERT RANKER: turn scored payments into alerts an investigator can work through.

Flagged payments are joined into one alert when they share an account within a few hours
(one money trail). Alerts are ranked by how suspicious and how large the trail is.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

LINK_WINDOW = 6 * 3600


def build_alerts(df, scores, top_frac=0.015):
    k = max(1, int(len(df) * top_frac))
    flagged = np.argsort(-scores, kind="stable")[:k]
    sub = df.iloc[flagged].assign(score=scores[flagged], row=flagged)
    # link flagged payments touching the same account within LINK_WINDOW
    ev = pd.concat([sub[["row", "si", "t"]].rename(columns={"si": "acc"}),
                    sub[["row", "ri", "t"]].rename(columns={"ri": "acc"})]).sort_values(["acc", "t"])
    same = (ev.acc.to_numpy()[1:] == ev.acc.to_numpy()[:-1]) & \
           (np.diff(ev.t.to_numpy()) <= LINK_WINDOW)
    pos = pd.Series(np.arange(k), index=sub.row)
    a = pos[ev.row.to_numpy()[:-1][same]].to_numpy()
    b = pos[ev.row.to_numpy()[1:][same]].to_numpy()
    _, comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(k, k)), directed=False)
    sub["alert"] = comp

    alerts = []
    for _, g in sub.groupby("alert"):
        g = g.sort_values("t")
        accounts = pd.unique(np.r_[g.s.to_numpy(), g.r.to_numpy()])
        risk = 0.6 * g.score.max() + 0.4 * g.score.mean()
        priority = risk * (1 + np.log1p(len(g)) / 3) * (1 + np.log1p(g.usd.sum()) / 20)
        alerts.append({"payments": g.hash.tolist(), "rows": g.row.tolist(), "n_payments": len(g),
                       "n_accounts": len(accounts), "total_usd": float(g.usd.sum()),
                       "start": int(g.t.min()), "end": int(g.t.max()),
                       "max_score": float(g.score.max()), "mean_score": float(g.score.mean()),
                       "risk": float(risk), "priority": float(priority)})
    alerts.sort(key=lambda x: -x["priority"])
    for i, a in enumerate(alerts, 1):
        a["alert_id"] = f"A{i:03d}"
    return alerts
