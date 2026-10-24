"""Two-pass gradient-boosting scorer with honest (grouped) cross-validation.

Pass 1 scores each payment from its own features. Pass 2 adds how suspicious the surrounding
payments look (same accounts, next/previous hops), using pass-1 scores that were made out of fold.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from .features import context

SEEDS = (0, 1)


def _gbm(seed):
    return HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=30, l2_regularization=1.0,
                                          max_features=0.6, random_state=seed)


class Scorer:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        self.models = [_gbm(s).fit(X, y) for s in SEEDS]
        return self

    def predict(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)


def groups(labels):
    """Look-alike scenarios (same cohort or scenario) never sit on both sides of a split."""
    g = labels.cohort_id.astype("string").fillna("s" + labels.scenario_id.astype("string"))
    return g.fillna("h" + labels.TRANSACTION_HASH).to_numpy()


def rank01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def blend(p1, p2):
    """Average rank of the two passes, mapped back onto pass 2's score range."""
    r = (rank01(p1) + rank01(p2)) / 2
    return np.sort(p2)[np.round(rank01(r) * (len(p2) - 1)).astype(int)]


def cross_validate(df, X, y, g, n_splits=5):
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(X, y, g))
    oof1 = np.zeros(len(y))
    for tr, va in folds:
        oof1[va] = Scorer().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    X2 = pd.concat([X, context(df, oof1)], axis=1)
    oof2 = np.zeros(len(y))
    for tr, va in folds:
        oof2[va] = Scorer().fit(X2.iloc[tr], y[tr]).predict(X2.iloc[va])
    return oof1, oof2


def report(y, score, difficulty):
    rows = {"all": average_precision_score(y, score) * 100}
    for d in ("easy", "medium", "hard"):
        m = (difficulty == d) | (y == 0)
        rows[d] = average_precision_score(y[m], score[m]) * 100
    k = max(1, int(round(len(y) * 0.01)))
    top = np.argsort(-score, kind="stable")[:k]
    rows["top1%_precision"] = y[top].mean() * 100
    rows["top1%_recall"] = y[top].sum() / y.sum() * 100
    return pd.Series(rows).round(1)
