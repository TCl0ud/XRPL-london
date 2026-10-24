"""Fast per-key time-window lookups (counts, sums, nearest event) with searchsorted."""
import numpy as np

BIG = 10_000_000  # larger than any time span in a split (seconds)

MIN, HOUR, DAY = 60, 3600, 86400


class Events:
    """Events keyed by an integer (account or pair), sorted by (key, time)."""

    def __init__(self, key, t, v=None):
        v = np.ones(len(t)) if v is None else np.asarray(v, dtype=float)
        o = np.lexsort((t, key))
        self.pos = o                      # original row of each sorted event
        self.k = np.asarray(key)[o]
        self.t = np.asarray(t)[o]
        self.v = v[o]
        self.comp = self.k.astype(np.int64) * BIG + self.t
        self.cs = np.r_[0.0, np.cumsum(self.v)]
        self.cs2 = np.r_[0.0, np.cumsum(self.v ** 2)]
        self.n = len(o)

    def bounds(self, key, lo, hi, closed_lo=False):
        """Index range of events with lo < t <= hi (lo <= t if closed_lo)."""
        key = np.asarray(key, dtype=np.int64)
        i = np.searchsorted(self.comp, key * BIG + lo, "left" if closed_lo else "right")
        j = np.searchsorted(self.comp, key * BIG + hi, "right")
        return i, j

    def count(self, key, lo, hi, closed_lo=False):
        i, j = self.bounds(key, lo, hi, closed_lo)
        return (j - i).astype(float)

    def total(self, key, lo, hi, closed_lo=False):
        i, j = self.bounds(key, lo, hi, closed_lo)
        return self.cs[j] - self.cs[i]

    def stats(self, key, lo, hi, closed_lo=False):
        """count, sum, coefficient of variation of values in the window."""
        i, j = self.bounds(key, lo, hi, closed_lo)
        n = (j - i).astype(float)
        s = self.cs[j] - self.cs[i]
        s2 = self.cs2[j] - self.cs2[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / n
            var = np.maximum(s2 / n - mean ** 2, 0)
            cv = np.where(n > 1, np.sqrt(var) / np.maximum(mean, 1e-9), np.nan)
        return n, s, cv

    def next_after(self, key, t):
        """Gap and value of the key's first event strictly after t (nan if none)."""
        key = np.asarray(key, dtype=np.int64)
        i = np.searchsorted(self.comp, key * BIG + t, "right")
        ok = i < self.n
        ok[ok] = self.k[i[ok]] == key[ok]
        gap = np.full(len(key), np.nan)
        val = np.full(len(key), np.nan)
        gap[ok] = self.t[i[ok]] - np.asarray(t)[ok]
        val[ok] = self.v[i[ok]]
        return gap, val

    def prev_before(self, key, t):
        """Gap and value of the key's last event strictly before t (nan if none)."""
        key = np.asarray(key, dtype=np.int64)
        i = np.searchsorted(self.comp, key * BIG + t, "left") - 1
        ok = i >= 0
        ok[ok] = self.k[i[ok]] == key[ok]
        gap = np.full(len(key), np.nan)
        val = np.full(len(key), np.nan)
        gap[ok] = np.asarray(t)[ok] - self.t[i[ok]]
        val[ok] = self.v[i[ok]]
        return gap, val


def ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
