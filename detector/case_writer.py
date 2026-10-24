"""CASE SUMMARY WRITER: a plain-English report for each top alert, where every claim points
at the payments (transaction hashes) behind it. Template-based, no outside services.
"""
from datetime import timedelta

import numpy as np
import pandas as pd


def _dur(sec):
    return str(timedelta(seconds=int(sec)))


def _pattern(g, senders, receivers, middles, X):
    if X.loc[g.index, ["cycle2_1d", "cycle3_1d"]].to_numpy().max() > 0:
        return "round trip", "money leaves an account and comes back to it (possible fake trading)"
    if X.loc[g.index, "just_below"].mean() > 0.5 and len(g) >= 3:
        return "structuring", "many payments just under a round threshold (possible limit dodging)"
    if len(middles) >= 2 and len(senders) <= 2 and len(receivers) <= 2:
        return "split and recombine", "money is spread over several middle accounts, then gathered again"
    if len(middles) >= 1 and X.loc[g.index, "hops_total_2d"].max() >= 2:
        return "layering chain", "money is passed hop by hop through intermediary accounts"
    if len(senders) == 1 and len(receivers) >= 3:
        return "fan-out", "one account sends to many accounts"
    if len(receivers) == 1 and len(senders) >= 3:
        return "fan-in", "many accounts pay into one account"
    return "linked activity", "a cluster of high-risk payments between connected accounts"


def _short(a):
    return f"`{a[:8]}…`"


def write_cases(df, X, scores, alerts, top=10):
    lines = ["# Case summaries", "",
             "> A flag is not proof of wrongdoing. Each case lists the payments behind it so an "
             "investigator can check every claim. Scores come from a model trained on synthetic data.",
             "", "**Glossary.** *Mule*: an account that receives money and quickly passes most of it on. "
             "*Hop*: one payment in a trail. *Structuring*: splitting money into smaller payments to "
             "stay under limits. *Round trip*: money that comes back to where it started.", ""]
    cases = []
    for a in alerts[:top]:
        g = df.iloc[a["rows"]].sort_values("t")
        senders, receivers = set(g.s), set(g.r)
        middles = senders & receivers
        sources, sinks = senders - receivers, receivers - senders
        name, meaning = _pattern(g, sources, sinks, middles, X)
        Xg = X.loc[g.index]
        fwd = Xg.fwd_gap.replace([np.inf], np.nan).dropna()
        reasons = []
        if len(fwd) and (fwd <= 600).mean() > 0.5:
            reasons.append(f"intermediaries passed money on fast (median {_dur(fwd.median())} after receiving it)")
        if Xg.pair_n.max() <= 1:
            reasons.append("every sender→receiver pair is new: none of them had paid each other before this week")
        elif (Xg.pair_n <= 1).mean() > 0.5:
            reasons.append(f"{int((Xg.pair_n <= 1).sum())} of {len(g)} payments use a brand-new sender→receiver pair")
        if Xg.just_below.sum() >= 2:
            reasons.append(f"{int(Xg.just_below.sum())} payments sit just below a round threshold")
        if Xg.cycle2_1d.max() + Xg.cycle3_1d.max() > 0:
            reasons.append("money returns to the account it left within a day")
        if Xg.s_passthrough.max() > 0.8 or Xg.r_passthrough.max() > 0.8:
            reasons.append("at least one account passes on almost everything it receives (mule-like)")
        if Xg.hops_total_2d.max() >= 2:
            reasons.append(f"the trail runs through up to {int(Xg.hops_total_2d.max()) + 1} hops")
        reasons.append(f"model scores from {g.index.map(lambda i: scores[i]).min():.2f} "
                       f"to {a['max_score']:.2f}")
        case = {"alert_id": a["alert_id"], "pattern": name, "meaning": meaning,
                "n_payments": a["n_payments"], "n_accounts": a["n_accounts"],
                "total_usd": round(a["total_usd"], 2), "duration": _dur(a["end"] - a["start"]),
                "risk": round(a["risk"], 3), "sources": sorted(sources), "intermediaries": sorted(middles),
                "collectors": sorted(sinks), "reasons": reasons, "payments": a["payments"]}
        cases.append(case)
        lines += [f"## {a['alert_id']}: {name} (risk {a['risk']:.2f})", "",
                  f"**Pattern:** {meaning}.", "",
                  f"**Size:** {a['n_payments']} payments, {a['n_accounts']} accounts, "
                  f"${a['total_usd']:,.0f} over {case['duration']}.", "",
                  "**Who's who:** sources " + (", ".join(map(_short, sorted(sources))) or "none") +
                  " → intermediaries " + (", ".join(map(_short, sorted(middles))) or "none") +
                  " → collectors " + (", ".join(map(_short, sorted(sinks))) or "none"), "",
                  "**Why it was flagged:**", *[f"- {r}" for r in reasons], "",
                  "| time (+h) | from | to | USD | score | transaction |", "|---|---|---|---|---|---|"]
        for i, p in g.iterrows():
            lines.append(f"| {p.t / 3600:.2f} | {_short(p.s)} | {_short(p.r)} | {p.usd:,.2f} | "
                         f"{scores[i]:.2f} | `{p.hash}` |")
        lines.append("")
    return cases, "\n".join(lines)
