#!/usr/bin/env python3
"""
quick_flat_recency_with_odds.py

Extends the flat turf recency pattern test by adding SP odds
and computing flat-stake ROI.

Pattern (unchanged):
1) Flat turf races (going != 'Standard')
2) Ran within 5 days
3) Last run turf
4) Top 6 on last two starts

Data source: public.runners_raw
"""

import pandas as pd
import psycopg
import re
from pathlib import Path

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
OUTDIR = Path("output")

FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


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


def fetch(conn) -> pd.DataFrame:
    sql = """
    SELECT
        date        AS race_date,
        race_id,
        type        AS race_code,
        going,
        horse       AS horse_key,
        pos         AS pos_int,
        sp
    FROM public.runners_raw
    WHERE date IS NOT NULL
      AND race_id IS NOT NULL
      AND horse IS NOT NULL
      AND pos IS NOT NULL
    ORDER BY horse, date, race_id;
    """
    return pd.read_sql(sql, conn)


def main():
    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        df = fetch(conn)

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")
    df["going"] = df["going"].astype(str)
    df["race_code"] = df["race_code"].astype(str)

    # Odds
    df["sp_dec"] = df["sp"].apply(frac_to_dec)
    df = df.dropna(subset=["sp_dec"])

    # Sort for lag logic
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)

    g = df.groupby("horse_key", sort=False)

    # Lag features
    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)
    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)
    df["prev_going_1"] = g["going"].shift(1)

    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    # Strategy mask (UNCHANGED)
    mask = (
        (df["race_code"].str.lower() == "flat") &
        (df["going"] != "Standard") &
        (df["days_since_run"].between(0, 5)) &
        (df["prev_going_1"] != "Standard") &
        (df["prev_pos_1"] <= 6) &
        (df["prev_pos_2"] <= 6)
    )

    qual = df[mask].copy()
    qual["win"] = (qual["pos_int"] == 1).astype(int)

    # Profit per bet
    qual["profit"] = qual["win"] * (qual["sp_dec"] - 1.0) - (1 - qual["win"])

    # Overall metrics
    bets = len(qual)
    wins = int(qual["win"].sum())
    profit = qual["profit"].sum()
    roi = profit / bets if bets else 0.0

    summary = pd.DataFrame([{
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets if bets else 0.0,
        "avg_odds": qual["sp_dec"].mean(),
        "profit": profit,
        "roi": roi,
    }])

    # Odds band diagnostics
    bins = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1000.0]
    labels = ["<=2", "2-3", "3-5", "5-8", "8-12", "12+"]

    qual["odds_band"] = pd.cut(qual["sp_dec"], bins=bins, labels=labels, right=False)

    by_band = (
        qual.groupby("odds_band", observed=True)
        .agg(
            bets=("win", "size"),
            wins=("win", "sum"),
            avg_odds=("sp_dec", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    by_band["roi"] = by_band["profit"] / by_band["bets"]

    OUTDIR.mkdir(exist_ok=True)
    summary.to_csv(OUTDIR / "flat_recency_odds_summary.csv", sep=";", decimal=",", index=False)
    by_band.to_csv(OUTDIR / "flat_recency_odds_by_band.csv", sep=";", decimal=",", index=False)

    print("\n=== FLAT TURF RECENCY WITH ODDS ===")
    print(summary.to_string(index=False))
    print("\n--- By odds band ---")
    print(by_band.to_string(index=False))

    print("\nWrote:")
    print(f"- {OUTDIR / 'flat_recency_odds_summary.csv'}")
    print(f"- {OUTDIR / 'flat_recency_odds_by_band.csv'}")


if __name__ == "__main__":
    main()
