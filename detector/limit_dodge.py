"""LIMIT DODGE DETECTOR (structuring / smurfing): a large sum broken into many smaller payments,
often just under a reporting threshold, often spread over several receivers or a longer period,
and the mirror image where many small payments are collected into one account.
"""
import numpy as np
import pandas as pd

from .events import DAY, HOUR, Events, ratio

THRESHOLDS = np.array([1_000, 2_000, 3_000, 5_000, 10_000, 15_000, 20_000, 50_000])
WINDOWS = {"1h": HOUR, "6h": 6 * HOUR, "1d": DAY, "3d": 3 * DAY}


def _below_threshold(usd, band=0.12):
    """How far under the next threshold the amount sits (0 = just below, 1 = far)."""
    nxt = THRESHOLDS[np.minimum(np.searchsorted(THRESHOLDS, usd, "right"), len(THRESHOLDS) - 1)]
    gap = np.where(usd < nxt, (nxt - usd) / nxt, 1.0)
    return gap, (gap <= band).astype(float)


def features(df):
    s, r, t, usd = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy(), df.usd.to_numpy()
    gap, just_below = _below_threshold(usd)
    f = {"thr_gap": gap, "just_below": just_below}

    out = Events(s, t, usd)
    inc = Events(r, t, usd)
    out_below = Events(s, t, just_below)
    in_below = Events(r, t, just_below)
    # pair key: how many separate payments the same sender sent the same receiver
    pair = s.astype(np.int64) * (int(r.max()) + 1) + r
    pair_ev = Events(pair, t, usd)
    for name, w in WINDOWS.items():
        n, tot, cv = out.stats(s, t - w, t + w, closed_lo=True)
        f[f"split_n_{name}"] = n                       # sender's payments around this one
        f[f"split_total_ratio_{name}"] = ratio(tot, usd)
        f[f"split_cv_{name}"] = cv                     # similar amounts -> low cv
        f[f"split_below_{name}"] = out_below.total(s, t - w, t + w, closed_lo=True)
        n, tot, cv = inc.stats(r, t - w, t + w, closed_lo=True)
        f[f"collect_n_{name}"] = n                     # receiver collecting many payments
        f[f"collect_total_ratio_{name}"] = ratio(tot, usd)
        f[f"collect_cv_{name}"] = cv
        f[f"collect_below_{name}"] = in_below.total(r, t - w, t + w, closed_lo=True)
        f[f"pair_repeat_{name}"] = pair_ev.count(pair, t - w, t + w, closed_lo=True)

    # Amount close to the sender's other payments (same chunk size repeated)
    gp, vp = out.prev_before(s, t)
    gn, vn = out.next_after(s, t)
    f["prev_out_gap"], f["next_out_gap"] = gp, gn
    f["prev_out_same_amt"] = np.abs(1 - ratio(vp, usd))
    f["next_out_same_amt"] = np.abs(1 - ratio(vn, usd))
    gp, vp = inc.prev_before(r, t)
    f["prev_in_gap"] = gp
    f["prev_in_same_amt"] = np.abs(1 - ratio(vp, usd))
    return pd.DataFrame(f)
