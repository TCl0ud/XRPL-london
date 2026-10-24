"""MONEY TRAIL TRACKER: follow money forward (receiver passes it on) and backward (where the
sender got it from), and measure how many hops a trail runs through.

Each payment is a hop. Payment j continues payment i when j's sender is i's receiver, j happens
within a time window after i, and j's amount is a plausible share of i's amount.
"""
import numpy as np
import pandas as pd

from .events import DAY, HOUR, MIN, Events, ratio

FLOW_WINDOWS = {"10m": 10 * MIN, "1h": HOUR, "6h": 6 * HOUR, "1d": DAY, "3d": 3 * DAY}
MAX_CANDIDATES = 30


def _hop_candidates(df, window, lo_ratio=0.2, hi_ratio=1.05):
    """For each payment, the next-hop payments (receiver's outgoing within `window`)."""
    out = Events(df.si.to_numpy(), df.t.to_numpy(), df.usd.to_numpy())
    i0, j0 = out.bounds(df.ri.to_numpy(), df.t.to_numpy(), df.t.to_numpy() + window)
    usd = df.usd.to_numpy()
    nxt = []
    for i, (a, b) in enumerate(zip(i0, j0)):
        cand = out.pos[a:min(b, a + MAX_CANDIDATES)]
        if len(cand):
            r = usd[cand] / max(usd[i], 1e-9)
            cand = cand[(r >= lo_ratio) & (r <= hi_ratio)]
        nxt.append(cand)
    return nxt


def hop_depth(df, window):
    """Longest chain of hops downstream and upstream of each payment, plus trail size."""
    nxt = _hop_candidates(df, window)
    n = len(df)
    order = np.argsort(df.t.to_numpy(), kind="stable")
    down = np.zeros(n)
    for i in order[::-1]:                     # later payments first
        if len(nxt[i]):
            down[i] = 1 + down[nxt[i]].max()
    up = np.zeros(n)
    n_prev = np.zeros(n)
    for i in order:                           # earlier payments first
        for j in nxt[i]:
            up[j] = max(up[j], up[i] + 1)
            n_prev[j] += 1
    n_next = np.array([len(c) for c in nxt], dtype=float)
    return down, up, n_next, n_prev


def features(df):
    s, r, t, usd = df.si.to_numpy(), df.ri.to_numpy(), df.t.to_numpy(), df.usd.to_numpy()
    out = Events(s, t, usd)    # each account's outgoing payments
    inc = Events(r, t, usd)    # each account's incoming payments
    f = {}
    for name, w in FLOW_WINDOWS.items():
        # forward: does the receiver pass the money on soon after?
        f[f"fwd_n_{name}"] = out.count(r, t, t + w)
        f[f"fwd_share_{name}"] = ratio(out.total(r, t, t + w), usd)
        # backward: did the sender just receive about this much?
        f[f"back_n_{name}"] = inc.count(s, t - w, t - 1, closed_lo=True)
        f[f"back_share_{name}"] = ratio(inc.total(s, t - w, t - 1, closed_lo=True), usd)
    gap, val = out.next_after(r, t)
    f["fwd_gap"] = gap
    f["fwd_first_ratio"] = ratio(val, usd)
    gap, val = inc.prev_before(s, t)
    f["back_gap"] = gap
    f["back_last_ratio"] = ratio(val, usd)

    for name, w in {"1h": HOUR, "2d": 2 * DAY}.items():
        down, up, n_next, n_prev = hop_depth(df, w)
        f[f"hops_down_{name}"] = down
        f[f"hops_up_{name}"] = up
        f[f"hops_total_{name}"] = down + up
        f[f"next_hops_{name}"] = n_next
        f[f"prev_hops_{name}"] = n_prev
    return pd.DataFrame(f)
