"""FAKE TRADING DETECTOR (wash trading / round-tripping): money or assets that go out and come
back to where they started (A->B->A, A->B->C->A), back-and-forth trading between the same pair,
and currency conversions that move value around without a commercial reason.
"""
import numpy as np
import pandas as pd

from .events import DAY, HOUR, Events, ratio

CAP = 15


def _cycles(df, window):
    """Does the money return to the sender within `window`, in 2 or 3 hops?"""
    s, r, t = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy()
    out = Events(s, t)
    i0, j0 = out.bounds(r, t, t + window)
    c2 = np.zeros(len(df))
    c3 = np.zeros(len(df))
    for i in range(len(df)):
        hop1 = out.pos[i0[i]:min(j0[i], i0[i] + CAP)]
        if not len(hop1):
            continue
        if np.any(r[hop1] == s[i]):
            c2[i] = 1
        a, b = out.bounds(r[hop1], t[hop1], np.full(len(hop1), t[i] + window))
        for x, y in zip(a, b):
            if y > x and np.any(r[out.pos[x:min(y, x + CAP)]] == s[i]):
                c3[i] = 1
                break
    return c2, c3


def features(df):
    s, r, t, usd = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy(), df.usd.to_numpy()
    n_acc = int(max(s.max(), r.max())) + 1
    pair = s.astype(np.int64) * n_acc + r
    rev = r.astype(np.int64) * n_acc + s
    pairs = Events(pair, t, usd)
    f = {}
    pc = pd.Series(pair).value_counts()
    f["pair_n"] = pd.Series(pair).map(pc).to_numpy().astype(float)
    f["rev_pair_n"] = pd.Series(rev).map(pc).fillna(0).to_numpy()
    f["pair_rank"] = pd.Series(pair).groupby(pair).cumcount().to_numpy().astype(float)
    gap, val = pairs.next_after(rev, t)       # B pays A back after A paid B
    f["return_gap"] = gap
    f["return_ratio"] = ratio(val, usd)
    gap, val = pairs.prev_before(rev, t)      # this payment is itself a return
    f["is_return_gap"] = gap
    f["is_return_ratio"] = ratio(val, usd)
    for name, w in {"1h": HOUR, "1d": DAY}.items():
        f[f"rev_n_{name}"] = pairs.count(rev, t - w, t + w, closed_lo=True)
        c2, c3 = _cycles(df, w)
        f[f"cycle2_{name}"], f[f"cycle3_{name}"] = c2, c3

    # Conversions: value in vs value out, and how much an account converts
    f["conv"] = df.conv.astype(float).to_numpy()
    f["legs"] = df.legs.astype(float).to_numpy()
    f["same_asset"] = (df.src_asset == df.dst_asset).astype(float).to_numpy()
    f["cost_ratio"] = ratio(df.src_usd.to_numpy(), usd)
    conv = df.conv.astype(float).to_numpy()
    f["s_conv_frac"] = pd.Series(conv).groupby(s).transform("mean").to_numpy()
    f["r_conv_frac"] = pd.Series(conv).groupby(r).transform("mean").to_numpy()
    f["dst_xrp"] = (df.dst_asset == "XRP").astype(float).to_numpy()
    f["src_xrp"] = (df.src_asset == "XRP").astype(float).to_numpy()
    return pd.DataFrame(f)
