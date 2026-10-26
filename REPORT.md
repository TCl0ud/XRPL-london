# Decode & Detect: report

**Team:** Pookies of The Caribbean · **Team leader:** Owen Leung · **Members:** Owen Leung, Jesang Limbu, Hemayu Vikeekumar, Mohamed Mustafa Avoor

**Submitted code:** the code ZIP was built from commit `02bbcf8`. After a rebase the same code is
commit `bb4b1f9` on `main`; later commits change only documentation.

Our tool gives every payment on the synthetic ledger a suspicion score from 0 to 1, groups the
most suspicious payments into money trails, names likely money mules, ranks the trails as alerts,
and writes a plain-English case report for each top alert, citing the transaction hashes behind
every claim.

**Best leaderboard score: 67.8** (100 × average precision). Random guessing scores about 3 and the
organisers' starter notebook 33.5.

> **Important:** a flag is a reason to investigate, not evidence of a crime. This is an
> experimental prototype tested only on synthetic data. See [Limitations](#4-limitations).

## 1. Data sources

- **Only the organisers' synthetic dataset**: a fictional week of XRP, issued-currency and
  cross-currency payments, with labelled training data and unlabelled test data.
  Training: 40,131 payments, 1,250 suspicious (about 3%) from 169 planted scenarios.
  Test: 40,101 payments between 700 accounts.
- Payments are valued in US dollars with the organisers' hourly reference prices
  (`market_rates.csv`).
- **No external data, external labels, pre-trained models or outside services.** No LLM is called
  when the code runs.
- The dataset is not included in this repository. Download both packages from
  https://hackathon.ukcbt.org and put the files in `data/train/` (with `labels.csv`) and
  `data/test/`.

## 2. Methodology

The final score is a **weighted average rank of two independent detectors**, built separately
and run with different Python versions:

```
                 ┌─► team pipeline   (xrpl-detector/) ─── score ──────────────────┐  75%
payments ────────┤                                                               ├──► blend.py ──► outputs/final.csv
                 └─► Owen's detector (detector/, run.py) ─ score × relay penalty ┘  25%
```

Each detector also groups its own top payments into alerts and writes case reports.

### What separates laundering from legitimate look-alikes

The legitimate data contains payments with **the same shapes** as the planted laundering
(fan-out, fan-in, chains, cycles), so shape alone gives many false alarms. Two signals worked:

1. **Speed.** Planted money moves fast: about 1 minute per hop, with easy scenarios finished in
   about 5 minutes, against about 60 minutes for legitimate look-alikes.
2. **New relationships.** Legitimate commerce repeats the same sender→receiver pairs. Planted
   flows almost always use a pair that has never paid before.

### Team pipeline (`xrpl-detector/`)

```
payments ──► score every payment ──► top 1% ──► group into trails ──► rank alerts ──► case reports
             aml_ranker/                         trail_tracker.py      alert_ranker.py  case_writer.py
             (two-stage ranker                   (10-min bursts)                        + trail_graph.py
              + mule network
              + relay penalty)
```

- **About 220 features per payment**, computed from the split's own week: value and conversion
  legs, whether the sender→receiver pair is new, each account's weekly profile, bursts of
  activity from 1 minute to 2 days, money in just before and out just after, and **episodes**
  (an account's consecutive payments linked within 30 seconds to 1 day).
- **Stage 1:** gradient-boosting classifiers (scikit-learn `HistGradientBoostingClassifier`,
  3 seeds). **Stage 2** adds how suspicious each payment's episode-mates look to stage 1, so a
  planted pattern lifts all of its payments together. The two are blended by average rank.
- **Mule network signal** (`aml_ranker/mule.py`). For each payment, take the highest blend score
  among the *other* payments touching the same sender or receiver within ±1 hour, and mix its
  rank in at 20%. Payments in the same scheme pull each other up.
- **Relay look-alike penalty** (`aml_ranker/relay.py`). Each relay cohort hides about one planted
  scenario among 99 legitimate look-alikes. A classifier learns "this payment belongs to a relay
  cohort" (`p_relay`), and the score becomes score × (1 − 0.5 × `p_relay`). We almost never catch
  the planted relays anyway, so this mostly removes false alarms.
- **Alerts:** flag the top 1% of scores (about 400 payments, the site's review budget), group
  flagged payments linked within 10 minutes (chain, fan-out or fan-in), rank alerts by
  0.5 × average score + 0.3 × highest score + 0.2 × size, and write a report for the top 5.

More detail: [`xrpl-detector/README.md`](xrpl-detector/README.md) and [`TEAM_NOTES.md`](TEAM_NOTES.md).

### Owen's detector (`detector/`, `run.py`)

About 170 features per payment from four pattern detectors:

| Part | File | What it looks for |
|---|---|---|
| Money trail tracker | `detector/trail_tracker.py` | Does the receiver pass the money on (10 min … 3 days), did the sender just receive it, and how many hops long is the trail |
| Mule network finder | `detector/mule_finder.py` | Accounts that pass on most of what they receive, fast, to new counterparties; networks of accounts linked by fast forwards |
| Limit dodge detector | `detector/limit_dodge.py` | Many similar payments around one sender or into one receiver, amounts just under round thresholds |
| Fake trading detector | `detector/fake_trading.py` | Money returning to where it started (A→B→A, A→B→C→A), back-and-forth pairs, conversions |

- **Scoring** (`detector/model.py`): gradient boosting (2 seeds); a second pass adds how
  suspicious the neighbouring payments look; the score is the average rank of both passes.
- **Alerts** (`detector/alert_ranker.py`): the top 1.5% of payments are joined into alerts when
  they share an account within 6 hours, then ranked by risk and size.
- **Case reports** (`detector/case_writer.py`): pattern, who's who, why it was flagged and every
  payment hash, for the top 10 alerts. `detector/mule_finder.py` also lists the 50 most likely mules.

### The blend (`blend.py`)

1. Apply the team pipeline's relay penalty to Owen's detector too: score × (1 − 0.5 × `p_relay`),
   with `p_relay` read from the team pipeline's `outputs/predictions.csv`.
2. Turn each score into a percentile rank, and take the weighted average: **75% team pipeline,
   25% Owen's detector**.
3. Re-rank the result to 0–1 and write `outputs/final.csv`.

Rank averaging works because the two detectors were built independently and make different
mistakes; neither score's scale matters, only its order.

## 3. Evaluation

### Leaderboard (test data)

| Run | Score | Top-1% precision | Top-1% recall |
|---|---|---|---|
| Organisers' starter notebook | 33.5 | 62% | 20% |
| Team pipeline: single gradient-boosting model | 59.2 | 94% | 30% |
| Team pipeline: two-stage ranker | 65.4 | 91% | 29% |
| Team pipeline: blend of stage 1 and stage 2 (Python 3.9.6, scikit-learn 1.6.1) | 66.5 | – | – |
| Team pipeline: + mule network signal | 66.8 | 92% | 30% |
| Team pipeline: + relay penalty 0.5 | 67.0 | 92% | 29% |
| Owen's detector alone | 64.3 | – | – |
| Blend, equal weights | 67.3 | – | – |
| Blend, 0.65 / 0.35 | 67.6 | – | – |
| Blend, 0.75 / 0.25 | 67.7 | – | – |
| Blend, 0.75 / 0.25, with burst smoothing on the team pipeline | 67.8 | – | – |
| **Blend, 0.75 / 0.25, relay penalty on both, no smoothing (submitted)** | **67.8** | – | – |

"–" means the site's figure was not recorded. Burst smoothing (pulling each score towards its
10-minute burst's average) tied the submitted run on the blend but cost the team pipeline 0.8
alone (66.0 vs 66.8), so it is switched off (`SMOOTH_OWN_WEIGHT = 1.0` in `xrpl-detector/config.py`).

The top 1% holds about 400 payments, so it can contain at most about 32% of the ~1,250
suspicious payments. 29–30% recall is close to that ceiling.

### Cross-validation (training data)

5 folds **grouped by scenario and by relay cohort**, so payments from one planted scenario (and
its look-alikes) never appear in both the training and the checking part. Without this the
scores would be unrealistically high. Average precision × 100:

| | All | Easy (64% of suspicious) | Medium (18%) | Hard (18%) |
|---|---|---|---|---|
| Team pipeline, blend (Python 3.9.6) | 56.9 | 75.0 | 4.7 | 1.1 |
| Team pipeline, + mule network signal | 57.8 | 75.6 | 5.2 | 1.2 |
| Team pipeline, + relay penalty 0.5 | 58.3 | – | – | – |
| Owen's detector, pass 1 | 52.4 | 69.0 | 4.5 | 0.8 |
| Owen's detector, pass 2 | 54.8 | 70.5 | 4.7 | 1.4 |
| Owen's detector, final score | 55.3 | 71.9 | 5.1 | 1.0 |

The blend itself was tuned on the leaderboard, not in cross-validation, so it has no CV figure.
Recall by family (team pipeline): strongest on cycle (85%), chain and fan-out-in (79%); weakest on
the relay families (aggregate relay 4%, distributed relay 0%).

### Alert quality (team pipeline)

Measured on the training labels with out-of-fold scores from an earlier model (this compares grouping rules, not models):

| Grouping | Alerts | Alerts that are real cases | Alerts per real case | Purity |
|---|---|---|---|---|
| Top 1%, 3-minute bursts | 62 | 85% | 1.15 | 99% |
| **Top 1%, 10-minute bursts (used)** | 54 | 85% | **1.00** | 99% |
| Top 1%, 60-minute bursts | 21 | 95% | 1.00 | 85% |

With 10-minute bursts **no real case is split across alerts** and alerts almost never mix two
cases. Longer windows start merging different cases (purity falls).

## 4. Limitations

- **A flag is not proof.** About 8% of the team pipeline's top 1% are false positives. A score is
  a reason for a human to look, not a finding of money laundering or of any wrongdoing.
- **Synthetic data only.** Good performance here does not show the tool would detect real fraud
  or laundering. Real data would need new testing, and data-protection checks before use.
- **Medium and hard cases are mostly missed.** Medium cases are slow and resemble legitimate
  activity. Each relay cohort hides one planted scenario among 99 legitimate ones that match it on
  every feature we tried, so relay recall is about 0%. 25 hard payments are labelled
  independently of anything observable, by design.
- **Alerts come from each detector, not the blend.** The submitted score is the blend, but alerts
  and case reports are written by each detector from its own score (`outputs/summaries.md` and
  `outputs/new/cases.md`), so a payment's alert can differ slightly from its blended rank.
- **Lone payments are not alerted.** A highly scored payment with no linked flagged payment keeps
  its score but does not appear as an alert or case report.
- **The blend weights were picked on the leaderboard** (five uploads), so 67.8 is slightly
  optimistic for new data.
- **Results vary slightly between library versions** (the team pipeline scored 66.5 rather than
  66.8 with Python 3.14 and scikit-learn 1.9.1). Use the pinned versions in
  [`README.md`](README.md) to reproduce the submitted predictions exactly.
- **Possible bias:** the models favour fast, new-relationship activity. Legitimate businesses
  that genuinely work this way (new customers, fast settlement) are more likely to be flagged.
- **Not a compliance product.** This is an educational prototype. It is not certified, not legal
  or financial advice, and not validated against any regulatory requirement.

## 5. Reproducing the results

See [`README.md`](README.md) for the exact commands, Python and library versions and run times.
In total about 30 minutes on a laptop CPU. No GPU, internet connection, paid service or API key
is needed, and all models use fixed seeds.

## 6. What we built and what we used

- **Our work** (built during the hacking period): everything in `xrpl-detector/`, `detector/`,
  `run.py`, `blend.py`, `third_model.py`, this report and the slides.
- **Not ours:** the organisers' dataset, starter notebook and `payments.py`, and open-source
  libraries. Full list with licences in [`THIRD_PARTY.md`](THIRD_PARTY.md).
- **AI assistance:** Claude (Anthropic) via Claude Code helped write code and documentation,
  including this report. Prompts: [`chat-history/`](chat-history/README.md). No AI is used
  when the tool runs.
