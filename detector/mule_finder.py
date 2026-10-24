"""MULE NETWORK FINDER: profile every account and find networks of pass-through accounts.

A money mule receives funds and quickly sends most of them on, usually to counterparties it has
no history with. Account profiles are mapped back onto each payment for both its sender (s_)
and its receiver (r_). Mule networks are connected groups of accounts linked by fast forwards.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .events import HOUR, MIN, Events, ratio


def account_profiles(df):
    n_acc = int(max(df.si.max(), df.ri.max())) + 1
    s, r, t, usd = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy(), df.usd.to_numpy()
    out = Events(s, t, usd)

    gap, _ = out.next_after(r, t)          # how long the receiver holds each incoming payment
    gap = np.where(np.isnan(gap), 1e7, gap)
    pair = pd.Series(s).astype(np.int64) * n_acc + r
    pair_n = pair.map(pair.value_counts()).to_numpy()

    def per_acc(idx, vals, how):
        return pd.Series(vals).groupby(idx).agg(how).reindex(range(n_acc)).to_numpy()

    p = pd.DataFrame(index=range(n_acc))
    p["in_n"] = np.bincount(r, minlength=n_acc)
    p["out_n"] = np.bincount(s, minlength=n_acc)
    p["in_usd"] = np.bincount(r, usd, minlength=n_acc)
    p["out_usd"] = np.bincount(s, usd, minlength=n_acc)
    p["passthrough"] = ratio(np.minimum(p.in_usd, p.out_usd), np.maximum(p.in_usd, p.out_usd))
    p["net_flow"] = ratio(p.in_usd - p.out_usd, p.in_usd + p.out_usd)
    p["hold_median"] = per_acc(r, gap, "median")
    p["fwd_10m_frac"] = per_acc(r, gap <= 10 * MIN, "mean")
    p["fwd_1h_frac"] = per_acc(r, gap <= HOUR, "mean")
    p["n_senders"] = per_acc(r, s, "nunique")
    p["n_receivers"] = per_acc(s, r, "nunique")
    p["new_pair_out_frac"] = per_acc(s, pair_n == 1, "mean")
    p["new_pair_in_frac"] = per_acc(r, pair_n == 1, "mean")
    acc = np.r_[s, r]
    tt = np.r_[t, t]
    first, last = per_acc(acc, tt, "min"), per_acc(acc, tt, "max")
    p["span"] = last - first
    p["first_seen"], p["last_seen"] = first, last

    # Mule networks: accounts joined by payments the receiver forwarded within an hour
    fast = gap <= HOUR
    g = coo_matrix((np.ones(fast.sum()), (s[fast], r[fast])), shape=(n_acc, n_acc))
    _, comp = connected_components(g, directed=False)
    comp_size = np.bincount(comp)
    p["net_id"] = comp
    p["net_size"] = comp_size[comp]
    fast_per_comp = np.bincount(comp[s[fast]], minlength=len(comp_size))
    p["net_fast_payments"] = fast_per_comp[comp]
    return p.fillna({"fwd_10m_frac": 0, "fwd_1h_frac": 0, "new_pair_out_frac": 0, "new_pair_in_frac": 0})


def features(df):
    p = account_profiles(df)
    cols = [c for c in p.columns if c not in ("net_id", "first_seen", "last_seen")]
    f = {}
    for side, idx in (("s", df.si.to_numpy()), ("r", df.ri.to_numpy())):
        for c in cols:
            f[f"{side}_{c}"] = p[c].to_numpy()[idx]
        f[f"{side}_age"] = df.t.to_numpy() - p.first_seen.to_numpy()[idx]
        f[f"{side}_remaining"] = p.last_seen.to_numpy()[idx] - df.t.to_numpy()
    f["same_network"] = (p.net_id.to_numpy()[df.si] == p.net_id.to_numpy()[df.ri]).astype(float)
    return pd.DataFrame(f)


def suspected_mules(df, scores, top=50):
    """Accounts that both pass money through and touch high-scoring payments."""
    p = account_profiles(df)
    accounts = pd.Series(np.r_[df.s, df.r], index=np.r_[df.si, df.ri]).groupby(level=0).first()
    touch = pd.concat([pd.DataFrame({"acc": df.si, "score": scores}),
                       pd.DataFrame({"acc": df.ri, "score": scores})])
    agg = touch.groupby("acc").score.agg(["max", "mean", "count"])
    p = p.join(agg)
    p["mule_score"] = p["max"] * p.passthrough.fillna(0) * (0.5 + 0.5 * p.fwd_1h_frac)
    p = p[(p.in_n > 0) & (p.out_n > 0)].sort_values("mule_score", ascending=False).head(top)
    p.insert(0, "account", accounts.reindex(p.index).to_numpy())
    return p[["account", "mule_score", "max", "passthrough", "fwd_1h_frac", "hold_median",
              "in_n", "out_n", "in_usd", "out_usd", "n_senders", "n_receivers", "net_size"]]
