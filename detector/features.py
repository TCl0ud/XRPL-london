"""All payment features: the payment itself, activity bursts, and the four detectors."""
import numpy as np
import pandas as pd

from . import fake_trading, limit_dodge, mule_finder, trail_tracker
from .events import HOUR, MIN, Events

BURSTS = {"1m": MIN, "5m": 5 * MIN, "30m": 30 * MIN, "3h": 3 * HOUR, "12h": 12 * HOUR}


def basic(df):
    usd = df.usd.to_numpy()
    f = {"log_usd": np.log1p(usd),
         "log_src_usd": np.log1p(df.src_usd.to_numpy()),
         "cents": np.round(df.dst_units.to_numpy() * 100) % 100,
         "round_10": (np.abs(usd - np.round(usd, -1)) < 0.5).astype(float),
         "round_100": (np.abs(usd - np.round(usd, -2)) < 1).astype(float),
         "hour": df.hour.to_numpy().astype(float),
         "fee": df.fee.to_numpy()}
    s, r, t = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy()
    out, inc = Events(s, t), Events(r, t)
    anyev = Events(np.r_[s, r], np.r_[t, t])
    glob = Events(np.zeros(len(t), dtype=np.int64), t)
    z = np.zeros(len(t), dtype=np.int64)
    for name, w in BURSTS.items():
        f[f"s_out_burst_{name}"] = out.count(s, t - w, t + w, closed_lo=True)
        f[f"r_in_burst_{name}"] = inc.count(r, t - w, t + w, closed_lo=True)
        f[f"s_any_burst_{name}"] = anyev.count(s, t - w, t + w, closed_lo=True)
        f[f"r_any_burst_{name}"] = anyev.count(r, t - w, t + w, closed_lo=True)
        f[f"ledger_burst_{name}"] = glob.count(z, t - w, t + w, closed_lo=True)
    return pd.DataFrame(f)


def build(df):
    parts = [basic(df), trail_tracker.features(df), mule_finder.features(df),
             limit_dodge.features(df), fake_trading.features(df)]
    X = pd.concat(parts, axis=1)
    return X.astype("float32")


def context(df, scores):
    """Second pass: how suspicious are the payments around this one (same accounts, same trail)?"""
    s, r, t = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy()
    sc = np.asarray(scores, dtype=float)
    ev = Events(np.r_[s, r], np.r_[t, t], np.r_[sc, sc])
    ev4 = Events(np.r_[s, r], np.r_[t, t], np.r_[sc, sc] ** 4)
    f = {}
    for name, w in {"10m": 10 * MIN, "1h": HOUR, "12h": 12 * HOUR, "3d": 72 * HOUR}.items():
        for side, a in (("s", s), ("r", r)):
            n = ev.count(a, t - w, t + w, closed_lo=True) - 1
            tot = ev.total(a, t - w, t + w, closed_lo=True) - sc
            p4 = ev4.total(a, t - w, t + w, closed_lo=True) - sc ** 4
            with np.errstate(invalid="ignore", divide="ignore"):
                f[f"ctx_{side}_mean_{name}"] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
                f[f"ctx_{side}_soft_max_{name}"] = np.where(n > 0, np.maximum(p4, 0) ** 0.25, np.nan)
            f[f"ctx_{side}_n_{name}"] = n
    for name, w in {"1h": HOUR, "2d": 48 * HOUR}.items():
        nxt = trail_tracker._hop_candidates(df, w)
        best_next = np.array([sc[c].max() if len(c) else np.nan for c in nxt])
        best_prev = np.full(len(df), np.nan)
        for i, c in enumerate(nxt):
            if len(c):
                best_prev[c] = np.fmax(best_prev[c], sc[i])
        f[f"ctx_next_hop_{name}"] = best_next
        f[f"ctx_prev_hop_{name}"] = best_prev
    f["ctx_self"] = sc
    return pd.DataFrame(f).astype("float32")
