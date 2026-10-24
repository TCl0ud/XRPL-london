# xrpl-owen-hackathon

A *Decode & Detect* entry. It gives every payment on the synthetic XRPL ledger a suspicion score
from 0 to 1, groups the riskiest payments into alerts, names likely money mules and writes a
case report for each top alert.

## Reproduce the predictions CSV

Python 3.14 (tested on 3.14.9, macOS, Apple A18 Pro, 8 GB RAM). Unzip the organisers'
training package and test package, then from this folder:

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --train path/to/decode-and-detect-training-data/train \
                        --test  path/to/decode-and-detect-test-data
```

(Defaults are `--train data/train --test data/test`.) The upload file is
**`outputs/predictions.csv`** (`TRANSACTION_HASH,score`, one row per test transaction).
The whole run takes about 8 minutes on a laptop CPU, most of it cross-validation; add
`--skip-cv` to skip the printed report. Results are deterministic (fixed seeds).

Other outputs in `outputs/`: `cv_report.txt`, `alerts.json`, `mules.csv`, `cases.md`, `cases.json`.

## How it works

| Part | File | What it looks for |
|---|---|---|
| Money trail tracker | `detector/trail_tracker.py` | Does the receiver pass the money on (10 min … 3 days), did the sender just receive it, and how many hops long is the trail |
| Mule network finder | `detector/mule_finder.py` | Accounts that pass on most of what they receive, fast, to new counterparties; networks of accounts linked by fast forwards |
| Limit dodge detector | `detector/limit_dodge.py` | Many similar payments around one sender or into one receiver, amounts just under round thresholds |
| Fake trading detector | `detector/fake_trading.py` | Money returning to where it started (A→B→A, A→B→C→A), back-and-forth pairs, conversions |
| Scoring model | `detector/model.py` | Gradient boosting (2 seeds) on ~170 features, then a second pass that adds how suspicious the neighbouring payments look; final score = average rank of both passes |
| Alert ranker | `detector/alert_ranker.py` | Joins the top 1.5% of payments into alerts when they share an account within 6 hours; ranks by risk and size |
| Case summary writer | `detector/case_writer.py` | A plain-English report per top alert: pattern, who's who, why it was flagged, every payment hash |

Validation is 5-fold cross-validation **grouped by scenario and cohort**, so look-alike
scenarios never appear in both training and validation (otherwise the score is too optimistic).

## Cross-validation result (training week, AP × 100)

| | All | Easy | Medium | Hard | Top-1% precision | Top-1% recall |
|---|---|---|---|---|---|---|
| Pass 1 | 52.4 | 69.0 | 4.5 | 0.8 | 78.3 | 25.1 |
| Pass 2 | 54.8 | 70.5 | 4.7 | 1.4 | 75.3 | 24.2 |
| **Blend (submitted)** | **55.3** | 71.9 | 5.1 | 1.0 | 76.8 | 24.6 |

## Licence and credits

MIT, see [`LICENSE`](LICENSE). What we did not write ourselves: [`THIRD_PARTY.md`](THIRD_PARTY.md).
