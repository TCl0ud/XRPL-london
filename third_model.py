"""Third model for the blend: Extra Trees on Owen's detector features (a different algorithm
makes different mistakes, which is what helps a rank blend).

    python third_model.py --train data/train --test data/test   # -> outputs/new/extra_trees.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from detector.data import load_split
from detector.features import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--test", default="data/test")
    ap.add_argument("--out", default="outputs/new/extra_trees.csv")
    a = ap.parse_args()
    train = load_split(a.train)
    y = train[["hash"]].merge(pd.read_csv(Path(a.train) / "labels.csv"), left_on="hash",
                              right_on="TRANSACTION_HASH", how="left").is_suspicious.to_numpy()
    X = build(train).fillna(-1)
    test = load_split(a.test)
    Xt = build(test).fillna(-1)
    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3,
                              class_weight="balanced_subsample", n_jobs=-1, random_state=0).fit(X, y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": et.predict_proba(Xt)[:, 1]}).to_csv(a.out, index=False)
    print(f"{len(test):,} rows -> {a.out}")
