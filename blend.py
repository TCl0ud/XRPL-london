"""Combine several prediction CSVs by weighted average rank.

    python blend.py --out outputs/final.csv --inputs outputs/submission.csv outputs/new/predictions.csv --weights 0.5 0.5
"""
import argparse

import pandas as pd


def blend(paths, weights):
    frames = [pd.read_csv(p).set_index("TRANSACTION_HASH")["score"] for p in paths]
    hashes = frames[0].index
    for f in frames[1:]:
        assert set(f.index) == set(hashes), "input files cover different transactions"
    r = sum(w * f.reindex(hashes).rank(pct=True) for f, w in zip(frames, weights)) / sum(weights)
    score = (r.rank(method="first") - 1) / (len(r) - 1)
    return pd.DataFrame({"TRANSACTION_HASH": hashes, "score": score.to_numpy()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = blend(a.inputs, a.weights or [1.0] * len(a.inputs))
    out.to_csv(a.out, index=False)
    print(f"{len(out):,} rows -> {a.out}")
