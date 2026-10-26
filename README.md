# Pookies of The Caribbean — XRPL hackathon entry

Our *Decode & Detect* entry (XRPL AI Hack, London, 24 October 2026; leaderboard **67.8**). It
gives every payment on the synthetic XRPL ledger a suspicion score from 0 to 1, groups the
riskiest payments into alerts, names likely money mules and writes a case report for each top alert.

- **How it works and how well:** [`REPORT.md`](REPORT.md) (data, method, leaderboard and
  cross-validation results, limitations)
- **Team pipeline:** [`xrpl-detector/README.md`](xrpl-detector/README.md); the team's handover
  notes are in [`TEAM_NOTES.md`](TEAM_NOTES.md)
- **AI use:** [`chat-history/`](chat-history/README.md)

## Team

**Pookies of The Caribbean**

| | |
|---|---|
| Team leader | Owen Leung |
| Members | Jesang Limbu, Hemayu Vikeekumar, Mohamed Mustafa Avoor |

## Results

The final score blends two independent detectors by weighted average rank (75% team pipeline,
25% Owen's detector):

| System | Folder | Leaderboard alone |
|---|---|---|
| Team pipeline (two-stage ranker + mule network signal + relay look-alike penalty) | `xrpl-detector/` | 67.0 (66.8 without the relay penalty) |
| Owen's detector (trails, mules, limit dodging, fake trading) | `detector/`, `run.py` | 64.3 |
| Average rank, equal weights (`blend.py`) | | 67.3 |
| Weighted average rank, 0.65 / 0.35 | | 67.6 |
| Weighted average rank, 0.75 / 0.25, before burst smoothing | | 67.7 |
| Weighted average rank, 0.75 / 0.25, with burst smoothing | | 67.8 |
| **Weighted average rank, 0.75 / 0.25, with relay penalty, no smoothing (`blend.py`, submitted)** | | **67.8** |

Random guessing scores about 3 and the organisers' starter notebook 33.5. The full history of
uploads is in [`REPORT.md`](REPORT.md#3-evaluation).

## Reproduce the predictions CSV

Download both data packages from https://hackathon.ukcbt.org and put the files in `data/train/`
(`transactions.parquet`, `transfers.parquet`, `market_rates.csv`, `labels.csv`) and `data/test/`
(the same files, without labels). The data is not in this repository.

The two systems were run with different Python versions, so each gets its own environment. Use
these exact versions to reproduce the submitted file: the team pipeline scores 66.5 rather than
66.8 with newer libraries (Python 3.14, scikit-learn 1.9.1).

| Environment | Python | Libraries (pinned) |
|---|---|---|
| Team pipeline | 3.9.6 (macOS system `python3`) | `xrpl-detector/requirements.txt`: pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, scipy 1.13.1, pyarrow 21.0.0, networkx 3.2.1, matplotlib 3.9.4, joblib 1.5.3 |
| Owen's detector and blend | 3.14.9 | `requirements.txt`: pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, pyarrow 26.0.0 |

```sh
# 1. Team pipeline: ~20 min training + ~1 min scoring
python3 -m venv .venv39
.venv39/bin/pip install -r xrpl-detector/requirements.txt
.venv39/bin/python xrpl-detector/train_model.py
.venv39/bin/python xrpl-detector/run_pipeline.py           # -> outputs/submission.csv, outputs/predictions.csv

# 2. Owen's detector: ~8 min
python3.14 -m venv .venv314
.venv314/bin/pip install -r requirements.txt
.venv314/bin/python run.py                                  # -> outputs/new/predictions.csv

# 3. Blend (submitted): relay penalty on Owen's detector, then 75% / 25% by rank
.venv314/bin/python blend.py --inputs outputs/submission.csv outputs/new/predictions.csv \
                             --weights 0.75 0.25 --relay-penalty 0.5 --out outputs/final.csv
```

**Upload `outputs/final.csv`** (`TRANSACTION_HASH,score`, one row per test payment, 40,101 rows).
Step 3 reads `p_relay` from the team pipeline's `outputs/predictions.csv`, so run step 1 first.

Measured on Apple Silicon Macs (M4 16 GB and A18 Pro 8 GB), CPU only. All models use fixed seeds.
No GPU, internet connection, paid service or API key is needed.

Other variants we tried (not submitted):

```sh
# Blend without the relay penalty on Owen's detector
.venv314/bin/python blend.py --inputs outputs/submission.csv outputs/new/predictions.csv \
                             --weights 0.75 0.25 --out outputs/final_no_relay.csv
# Three models: adds Extra Trees on Owen's detector features (~15 s)
.venv314/bin/python third_model.py                           # -> outputs/new/extra_trees.csv
.venv314/bin/python blend.py --inputs outputs/submission.csv outputs/new/predictions.csv outputs/new/extra_trees.csv \
                             --weights 0.70 0.20 0.10 --out outputs/final_3models.csv
```

Burst smoothing in the team pipeline (`xrpl-detector/predict.py`) is switched off
(`SMOOTH_OWN_WEIGHT = 1.0` in `xrpl-detector/config.py`): at 0.5 the team pipeline alone scored
66.0 vs 66.8 without it.

## What each run writes

| File | Written by | What it is |
|---|---|---|
| `outputs/final.csv` | `blend.py` | **The upload file** |
| `outputs/submission.csv` | team pipeline | Team pipeline score in the upload format (67.0 alone); `submission_*.csv` are the variants compared in [`REPORT.md`](REPORT.md) |
| `outputs/predictions.csv` | team pipeline | One row per payment with every feature, `probability` and `p_relay` |
| `outputs/alerts.json`, `ranked_alerts.json` | team pipeline | Money trails from the top 1% of scores, then ranked |
| `outputs/summaries.md`, `graphs/` | team pipeline | Case reports for the top 5 alerts, with trail graphs |
| `outputs/new/predictions.csv` | Owen's detector | Owen's detector score in the upload format (64.3 alone) |
| `outputs/new/cv_report.txt` | Owen's detector | Cross-validation results |
| `outputs/new/alerts.json`, `mules.csv` | Owen's detector | Alerts from the top 1.5% of scores, and the 50 most likely mules |
| `outputs/new/cases.md`, `cases.json` | Owen's detector | Case reports for the top 10 alerts |

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

Cross-validation is 5-fold, **grouped by scenario and cohort**, so look-alike scenarios never sit
on both sides of a split (average precision × 100):

| | All | Easy | Medium | Hard | Top-1% precision | Top-1% recall |
|---|---|---|---|---|---|---|
| Pass 1 | 52.4 | 69.0 | 4.5 | 0.8 | 78.3 | 25.1 |
| Pass 2 | 54.8 | 70.5 | 4.7 | 1.4 | 75.3 | 24.2 |
| **Detector score** | **55.3** | 71.9 | 5.1 | 1.0 | 76.8 | 24.6 |

Add `--skip-cv` to `run.py` to skip the cross-validation report (faster; the predictions are the same).

## Team pipeline (`xrpl-detector/`)

See [`xrpl-detector/README.md`](xrpl-detector/README.md) for how it works and how to run it on its
own, and [`TEAM_NOTES.md`](TEAM_NOTES.md) for the team's handover notes and experiments. A cleaned-up
copy of the team pipeline alone is at https://github.com/jesanglimbu/AML-Model-XRPL.

## Repository layout

| Path | What it is |
|---|---|
| `xrpl-detector/` | Team pipeline: scoring (`aml_ranker/`), trails, alert ranking, graphs, case reports |
| `detector/`, `run.py` | Owen's detector |
| `blend.py` | Combines prediction files by weighted average rank, with the optional relay penalty |
| `third_model.py` | Extra Trees on Owen's detector features (three-model variant, not submitted) |
| `REPORT.md` | Method, results and limitations |
| `TEAM_NOTES.md` | The team's handover notes from the day |
| `chat-history/` | Each team member's AI prompts, for the competition terms |
| `LICENSE`, `THIRD_PARTY.md` | MIT licence and what we did not write ourselves |

## Licence and credits

MIT, see [`LICENSE`](LICENSE). What we did not write ourselves: [`THIRD_PARTY.md`](THIRD_PARTY.md).
Code and documentation were written with help from Claude (Anthropic) via Claude Code; no AI is
used when the code runs.
