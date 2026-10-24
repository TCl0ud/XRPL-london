"""Train on the labelled week, score the test week, and write every output.

    python run.py --train data/train --test data/test

Writes to outputs/: predictions.csv (upload this), cv_report.txt, alerts.json, mules.csv,
cases.md and cases.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from detector.alert_ranker import build_alerts
from detector.case_writer import write_cases
from detector.data import load_split
from detector.features import build, context
from detector.model import Scorer, blend, cross_validate, groups, report
from detector.mule_finder import suspected_mules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--test", default="data/test")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--skip-cv", action="store_true", help="skip cross-validation (faster)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train = load_split(a.train)
    lab = train[["hash"]].merge(pd.read_csv(Path(a.train) / "labels.csv"),
                                left_on="hash", right_on="TRANSACTION_HASH", how="left")
    y = lab.is_suspicious.to_numpy()
    X = build(train)
    print(f"train: {len(y):,} payments, {y.sum():,} suspicious, {X.shape[1]} features")

    if not a.skip_cv:
        oof1, oof2 = cross_validate(train, X, y, groups(lab))
        text = []
        for name, s in (("pass 1", oof1), ("pass 2", oof2), ("blend (submitted)", blend(oof1, oof2))):
            text.append(f"{name}:\n{report(y, s, lab.difficulty.to_numpy()).to_string()}\n")
        print("\n".join(text))
        (out / "cv_report.txt").write_text("5-fold CV grouped by scenario/cohort (AP x 100)\n\n" + "\n".join(text))
    else:
        # pass-2 training needs out-of-fold pass-1 scores; get them without the full report
        oof1, _ = cross_validate(train, X, y, groups(lab))

    m1 = Scorer().fit(X, y)
    m2 = Scorer().fit(pd.concat([X, context(train, oof1)], axis=1), y)

    test = load_split(a.test)
    Xt = build(test)
    p1 = m1.predict(Xt)
    p2 = m2.predict(pd.concat([Xt, context(test, p1)], axis=1))
    score = np.clip(blend(p1, p2), 0, 1)

    sub = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": score})
    assert sub.TRANSACTION_HASH.is_unique and len(sub) == len(test)
    sub.to_csv(out / "predictions.csv", index=False)

    alerts = build_alerts(test, score)
    json.dump([{k: v for k, v in x.items() if k != "rows"} for x in alerts],
              open(out / "alerts.json", "w"), indent=1)
    suspected_mules(test, score).to_csv(out / "mules.csv", index=False)
    cases, md = write_cases(test, Xt, score, alerts)
    (out / "cases.md").write_text(md)
    json.dump(cases, open(out / "cases.json", "w"), indent=1)

    print(f"{len(sub):,} test payments scored -> {out / 'predictions.csv'}")
    print(f"{len(alerts)} alerts, top 10 written up in {out / 'cases.md'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
