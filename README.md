# xrpl-owen-hackathon

A *Decode & Detect* entry (leaderboard **67.8**). It gives every payment on the synthetic XRPL
ledger a suspicion score from 0 to 1, groups the riskiest payments into alerts, names likely money
mules and writes a case report for each top alert.

The final score blends two independent detectors by weighted average rank (75% team pipeline, 25% Owen's detector):

| System | Folder | Leaderboard alone |
|---|---|---|
| Team pipeline (two-stage ranker + mule network signal + burst smoothing) | `xrpl-detector/` | 66.8 before smoothing |
| Owen's detector (trails, mules, limit dodging, fake trading) | `detector/`, `run.py` | 64.3 |
| Average rank, equal weights (`blend.py`) | | 67.3 |
| Weighted average rank, 0.65 / 0.35 | | 67.6 |
| Weighted average rank, 0.75 / 0.25, before burst smoothing | | 67.7 |
| **Weighted average rank, 0.75 / 0.25, with burst smoothing (`blend.py`, submitted)** | | **67.8** |

## Reproduce the predictions CSV

Put the organisers' data in `data/train/` (with `labels.csv`) and `data/test/` in this folder.
The two systems were run with different Python versions, so each gets its own environment.

```sh
# 1. Team pipeline: Python 3.9.6 (macOS system python3), ~20 min training + ~1 min scoring
python3 -m venv .venv39
.venv39/bin/pip install -r xrpl-detector/requirements.txt
.venv39/bin/python xrpl-detector/train_model.py
.venv39/bin/python xrpl-detector/run_pipeline.py           # -> outputs/submission.csv

# 2. Owen's detector: Python 3.14.9, ~8 min
python3.14 -m venv .venv314
.venv314/bin/pip install -r requirements.txt
.venv314/bin/python run.py                                  # -> outputs/new/predictions.csv

# 3. Blend (75% team pipeline, 25% Owen's detector)
.venv314/bin/python blend.py --inputs outputs/submission.csv outputs/new/predictions.csv \
                             --weights 0.75 0.25 --out outputs/final.csv
```

Burst smoothing (`xrpl-detector/predict.py`, `SMOOTH_OWN_WEIGHT = 0.5` in `xrpl-detector/config.py`) pulls each
payment's score halfway towards the average of its 3- and 10-minute burst.

The upload file is **`outputs/final.csv`** (`TRANSACTION_HASH,score`, one row per test transaction).
Measured on Apple Silicon Macs (M4 16 GB and A18 Pro 8 GB), CPU only. Both systems use fixed
seeds. The team pipeline scored 66.5 rather than 66.8 with newer libraries (Python 3.14,
scikit-learn 1.9.1), so use the pinned versions.

## Owen's detector (`detector/`)

| Part | File | What it looks for |
|---|---|---|
| Money trail tracker | `detector/trail_tracker.py` | Does the receiver pass the money on (10 min … 3 days), did the sender just receive it, and how many hops long is the trail |
| Mule network finder | `detector/mule_finder.py` | Accounts that pass on most of what they receive, fast, to new counterparties; networks of accounts linked by fast forwards |
| Limit dodge detector | `detector/limit_dodge.py` | Many similar payments around one sender or into one receiver, amounts just under round thresholds |
| Fake trading detector | `detector/fake_trading.py` | Money returning to where it started (A→B→A, A→B→C→A), back-and-forth pairs, conversions |
| Scoring model | `detector/model.py` | Gradient boosting (2 seeds) on ~170 features, then a second pass that adds how suspicious the neighbouring payments look; score = average rank of both passes |
| Alert ranker | `detector/alert_ranker.py` | Joins the top 1.5% of payments into alerts when they share an account within 6 hours; ranks by risk and size |
| Case summary writer | `detector/case_writer.py` | A plain-English report per top alert: pattern, who's who, why it was flagged, every payment hash |

`run.py` also writes `outputs/new/alerts.json`, `mules.csv`, `cases.md` and `cases.json`.
Cross-validation is 5-fold, **grouped by scenario and cohort**, so look-alike scenarios never sit
on both sides of a split:

| | All | Easy | Medium | Hard | Top-1% precision | Top-1% recall |
|---|---|---|---|---|---|---|
| Pass 1 | 52.4 | 69.0 | 4.5 | 0.8 | 78.3 | 25.1 |
| Pass 2 | 54.8 | 70.5 | 4.7 | 1.4 | 75.3 | 24.2 |
| **Detector score** | **55.3** | 71.9 | 5.1 | 1.0 | 76.8 | 24.6 |

## Team pipeline (`xrpl-detector/`)

See [`xrpl-detector/README.md`](xrpl-detector/README.md) and the team's notes in
[`TEAM_NOTES.md`](TEAM_NOTES.md). It also writes trail alerts, ranked alerts, trail graphs and
case summaries to `outputs/`.

## Licence and credits

MIT, see [`LICENSE`](LICENSE). What we did not write ourselves: [`THIRD_PARTY.md`](THIRD_PARTY.md).
