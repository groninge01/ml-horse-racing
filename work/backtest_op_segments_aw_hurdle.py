#!/usr/bin/env python3
"""
backtest_op_segments_aw_hurdle.py

Backtest OP-settle-OP (fixed odds) for:
- Flat turf handicaps
- Flat AW handicaps (going == 'Standard')
- Hurdle handicaps

Qualifier:
- ran within 5 days
- top-6 finish on last 2 starts
- plus segment-specific "same surface/code" consistency

Betting:
- bet @ OP (from comment)
- settle @ OP
- no SP settlement, no BOG assumptions

Writes:
- output/op_settle_op_segments_summary.csv
- output/op_settle_op_segments_by_oddsband.csv
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import psycopg

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
OUTDIR = Path("output")

FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
OP_RE = re.compile(r"\bop\s+(\d+\s*/\s*\d+)\b", re.IGNORECASE)


def frac_to_dec(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"", "-", "nan"}:
        return None
    if s in {"evens", "evs"}:
        return 2.0
    m = FRAC_RE.search(s)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return None
    return 1.0 + a / b


def extract_op_frac(comment):
    if comment is None:
        return None
    s = str(comment).lower()
    m = OP_RE.search(s)
    if not m:
        return None
    return m.group(1).replace(" ", "")


def fetch(conn) -> pd.DataFrame:
    sql = """
    SELECT
        date        AS race_date,
        race_id,
        race_name,
        type        AS race_type,
        going,
        horse       AS horse_key,
        pos         AS pos_int,
        sp,
        comment
    FROM public.runners_raw
    WHERE date IS NOT NULL
      AND race_id IS NOT NULL
      AND horse IS NOT NULL
      AND pos IS NOT NULL
      AND comment IS NOT NULL
    ORDER BY horse, date, race_id;
    """
    return pd.read_sql(sql, conn)


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")
    df["race_type"] = df["race_type"].astype(str)
    df["going"] = df["going"].astype(str)
    df["race_name"] = df["race_name"].astype(str)

    # OP
    df["op_frac"] = df["comment"].apply(extract_op_frac)
    df["op_dec"] = df["op_frac"].apply(frac_to_dec)
    df = df.dropna(subset=["op_dec", "race_date", "pos_int"]).copy()

    # Handicap
    df["handicap"] = df["race_name"].str.contains("handicap", case=False, na=False)

    # Sort & lag
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_key", sort=False)

    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)
    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)
    df["prev_going_1"] = g["going"].shift(1)
    df["prev_type_1"] = g["race_type"].shift(1)
    df["prev_type_2"] = g["race_type"].shift(2)

    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    return df


def odds_band_series(op_dec: pd.Series) -> pd.Series:
    bins = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1000.0]
    labels = ["<=2", "2-3", "3-5", "5-8", "8-12", "12+"]
    return pd.cut(op_dec, bins=bins, labels=labels, right=False)


def backtest_segment(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df must already be filtered to the segment universe.
    Applies qualifier + OP settle OP PnL.
    """
    d = df.copy()

    # Core qualifier: last 2 starts top6, within 5 days
    d["qualifier_core"] = (
        d["days_since_run"].between(0, 5) &
        (d["prev_pos_1"] <= 6) &
        (d["prev_pos_2"] <= 6)
    )

    # Segment-specific consistency rules
    if name == "flat_turf_handicap":
        # turf = not Standard, and last run also turf
        d = d[(d["going"] != "Standard") & (d["prev_going_1"] != "Standard")].copy()
        # keep last two starts also flat (race_type == 'Flat')
        d = d[(d["race_type"].str.lower() == "flat") &
              (d["prev_type_1"].str.lower() == "flat") &
              (d["prev_type_2"].str.lower() == "flat")].copy()

    elif name == "flat_aw_handicap":
        # AW = Standard, last run also AW
        d = d[(d["going"] == "Standard") & (d["prev_going_1"] == "Standard")].copy()
        d = d[(d["race_type"].str.lower() == "flat") &
              (d["prev_type_1"].str.lower() == "flat") &
              (d["prev_type_2"].str.lower() == "flat")].copy()

    elif name == "hurdle_handicap":
        # hurdles only, and last two also hurdles (avoid mixing)
        d = d[(d["race_type"].str.lower() == "hurdle") &
              (d["prev_type_1"].str.lower() == "hurdle") &
              (d["prev_type_2"].str.lower() == "hurdle")].copy()

    else:
        raise ValueError(f"Unknown segment: {name}")

    # Handicap only
    d = d[d["handicap"]].copy()

    # Apply qualifier
    bets = d[d["qualifier_core"]].copy()

    # Outcome + OP-settle profit
    bets["win"] = (bets["pos_int"] == 1).astype(int)
    bets["profit"] = bets["win"] * (bets["op_dec"] - 1.0) - (1 - bets["win"])

    # Summary
    n = len(bets)
    wins = int(bets["win"].sum())
    profit = float(bets["profit"].sum())
    roi = profit / n if n else 0.0

    summary = pd.DataFrame([{
        "segment": name,
        "bets": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "avg_op_odds": float(bets["op_dec"].mean()) if n else 0.0,
        "profit": profit,
        "roi": roi,
    }])

    # By odds band (based on OP)
    bets["odds_band"] = odds_band_series(bets["op_dec"])
    by_band = (
        bets.groupby("odds_band", observed=True)
        .agg(
            bets=("win", "size"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
            avg_op_odds=("op_dec", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    by_band["roi"] = by_band["profit"] / by_band["bets"]
    by_band.insert(0, "segment", name)

    return summary, by_band


def main():
    OUTDIR.mkdir(exist_ok=True)

    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        df = fetch(conn)

    df = add_lags(df)

    segments = [
        "flat_turf_handicap",
        "flat_aw_handicap",
        "hurdle_handicap",
    ]

    summaries = []
    bands = []

    for seg in segments:
        s, b = backtest_segment(df, seg)
        summaries.append(s)
        bands.append(b)

    summary_df = pd.concat(summaries, ignore_index=True)
    bands_df = pd.concat(bands, ignore_index=True)

    summary_path = OUTDIR / "op_settle_op_segments_summary.csv"
    bands_path = OUTDIR / "op_settle_op_segments_by_oddsband.csv"

    summary_df.to_csv(summary_path, sep=";", decimal=",", index=False)
    bands_df.to_csv(bands_path, sep=";", decimal=",", index=False)

    print("\n=== OP BACKTEST (BET @ OP, SETTLE @ OP) ===")
    print(summary_df.to_string(index=False))
    print("\nWrote:")
    print(f"- {summary_path}")
    print(f"- {bands_path}")


if __name__ == "__main__":
    main()
