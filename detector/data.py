"""Load one split into a table with one row per payment, valued in reference USD."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

XRP_SENTINEL = "0x1111222211112222111122221111222211112222"


def _decode(wire):
    """DELIVERED_AMOUNT JSON -> (asset key, amount in asset units)."""
    w = json.loads(wire)
    if isinstance(w, str):
        return "XRP", int(w) / 1e6
    return f"{w['issuer']}:{w['currency']}", float(w["value"])


def _usd(rates, ts, assets, units):
    hour = (np.asarray(ts, dtype=np.int64) // 3600) * 3600
    keys = pd.MultiIndex.from_arrays([hour, assets])
    price = rates.reindex(keys).to_numpy() / 1e6
    return np.asarray(units, dtype=float) * price


def load_split(directory):
    d = Path(directory)
    tx = pd.read_parquet(d / "transactions.parquet",
                         columns=["TRANSACTION_HASH", "TIMESTAMP", "ACCOUNT", "FEE",
                                  "DELIVERED_AMOUNT", "EXTRA_TRANSACTION_FIELDS"])
    tr = pd.read_parquet(d / "transfers.parquet",
                         columns=["TRANSACTION_HASH", "TRANSFER_INDEX", "TRANSFER_TYPE",
                                  "TOKEN_ADDRESS", "SYMBOL", "VALUE"])
    rates = pd.read_csv(d / "market_rates.csv").set_index(["TIMESTAMP", "ASSET"])["PRICE_MICRO_USD"]

    df = pd.DataFrame({"hash": tx.TRANSACTION_HASH.to_numpy(),
                       "t": tx.TIMESTAMP.astype(np.int64).to_numpy(),
                       "s": tx.ACCOUNT.to_numpy(),
                       "r": [json.loads(x)["Destination"] for x in tx.EXTRA_TRANSACTION_FIELDS],
                       "fee": tx.FEE.astype(float).to_numpy()})
    dec = [_decode(x) for x in tx.DELIVERED_AMOUNT]
    df["dst_asset"] = [a for a, _ in dec]
    df["dst_units"] = [u for _, u in dec]
    df["usd"] = _usd(rates, df.t, df.dst_asset, df.dst_units)

    # Actual source cost = first executed leg (SendMax is only a ceiling)
    tr["TRANSFER_INDEX"] = tr.TRANSFER_INDEX.astype(int)
    tr = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    first = tr.groupby("TRANSACTION_HASH").first()
    legs = tr.groupby("TRANSACTION_HASH").size()
    first_asset = np.where(first.TOKEN_ADDRESS == XRP_SENTINEL, "XRP",
                           first.TOKEN_ADDRESS + ":" + first.SYMBOL)
    src = pd.DataFrame({"src_asset": first_asset, "src_units": first.VALUE.to_numpy(),
                        "conv": (first.TRANSFER_TYPE == "path_payment").to_numpy()},
                       index=first.index)
    src["legs"] = legs
    df = df.join(src, on="hash")
    df["src_usd"] = _usd(rates, df.t, df.src_asset, df.src_units)
    df["usd"] = df.usd.fillna(df.src_usd).fillna(0.0)
    df["src_usd"] = df.src_usd.fillna(df.usd)

    df = df.sort_values(["t", "hash"], kind="stable").reset_index(drop=True)
    accounts = pd.Index(pd.unique(np.r_[df.s.to_numpy(), df.r.to_numpy()]))
    df["si"] = accounts.get_indexer(df.s)
    df["ri"] = accounts.get_indexer(df.r)
    df["hour"] = (df.t // 3600) % 24
    df["t"] = df.t - df.t.min()
    return df
