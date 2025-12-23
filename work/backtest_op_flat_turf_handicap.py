#!/usr/bin/env python3
"""
backtest_op_flat_turf_handicap_correct.py

Correct OP-only backtest for the recency+consistency strategy.

Universe:
- Flat
- Turf only (going != 'Standard')
- Handicap only (race_name contains 'handicap')

Qualifier:
- days_since_run in [0..5]
- prev_pos_1 <= 6 and prev_pos_2 <= 6
- previous run also turf (prev_going_1 != 'Standard')

Betting:
- BET at OP (fixed odds)
- SETTLE at OP (correct: no BOG)

Also outputs diagnostic about price movement OP -> SP (NOT used for settlement).

Writes EU-formatted CSVs to output/.
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


def fetch(conn):
    sql = """
    SELECT
        date        AS race_date,
        race_id,
        race_name,
        type        AS race_code,
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


def main():
    OUTDIR.mkdir(exist_ok=True)

    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        df = fetch(conn)

    # basic clean
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")
    df["race_code"] = df["race_code"].astype(str)
    df["going"] = df["going"].astype(str)
    df["race_name"] = df["race_name"].astype(str)

    # universe: flat turf handicap
    df = df[
        (df["race_code"].str.lower() == "flat") &
        (df["going"] != "Standard") &
        (df["race_name"].str.contains("handicap", case=False, na=False))
    ].copy()

    # odds
    df["sp_dec"] = df["sp"].apply(frac_to_dec)  # optional for diagnostics
    df["op_frac"] = df["comment"].apply(extract_op_frac)
    df["op_dec"] = df["op_frac"].apply(frac_to_dec)

    # require OP to bet
    df = df.dropna(subset=["op_dec"]).copy()

    # lags
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_key", sort=False)

    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)
    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)
    df["prev_going_1"] = g["going"].shift(1)
    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    # qualifier
    df["qualifier"] = (
        (df["days_since_run"].between(0, 5)) &
        (df["prev_pos_1"] <= 6) &
        (df["prev_pos_2"] <= 6) &
        (df["prev_going_1"] != "Standard")
    )

    bets = df[df["qualifier"]].copy()

    # outcomes
    bets["win"] = (bets["pos_int"] == 1).astype(int)

    # ✅ correct profit at OP (fixed odds settlement)
    bets["profit"] = bets["win"] * (bets["op_dec"] - 1.0) - (1 - bets["win"])

    # odds bands by OP
    bins = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1000.0]
    labels = ["<=2", "2-3", "3-5", "5-8", "8-12", "12+"]
    bets["odds_band"] = pd.cut(bets["op_dec"], bins=bins, labels=labels, right=False)

    # overall summary
    n = len(bets)
    wins = int(bets["win"].sum())
    profit = float(bets["profit"].sum())
    roi = profit / n if n else 0.0

    summary = pd.DataFrame([{
        "segment": "flat_turf_handicap_OP_settle_OP",
        "bets": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "avg_op_odds": float(bets["op_dec"].mean()) if n else 0.0,
        "profit": profit,
        "roi": roi,
    }])

    # by band summary
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
    by_band.insert(0, "segment", "flat_turf_handicap_OP_settle_OP")

    # -----------------------
    # Diagnostics: OP -> SP move
    # (NOT settlement; just insight)
    # -----------------------
    diag = bets.dropna(subset=["sp_dec"]).copy()
    if len(diag):
        diag["shortened"] = (diag["sp_dec"] < diag["op_dec"]).astype(int)  # price moved in your favour
        diag["move"] = diag["op_dec"] - diag["sp_dec"]  # positive = shortened
        move_by_band = (
            diag.groupby("odds_band", observed=True)
            .agg(
                runners=("shortened", "size"),
                shortened_rate=("shortened", "mean"),
                mean_move=("move", "mean"),
                median_move=("move", "median"),
                mean_move_winners=("move", lambda s: float(s[diag.loc[s.index, "win"] == 1].mean()) if (diag.loc[s.index, "win"] == 1).any() else float("nan")),
                mean_move_losers=("move", lambda s: float(s[diag.loc[s.index, "win"] == 0].mean()) if (diag.loc[s.index, "win"] == 0).any() else float("nan")),
            )
            .reset_index()
        )
    else:
        move_by_band = pd.DataFrame(columns=["odds_band","runners","shortened_rate","mean_move","median_move","mean_move_winners","mean_move_losers"])

    # write outputs (EU formatting)
    summary_path = OUTDIR / "op_settle_op_flat_turf_handicap_summary.csv"
    bands_path = OUTDIR / "op_settle_op_flat_turf_handicap_by_oddsband.csv"
    move_path = OUTDIR / "op_to_sp_move_flat_turf_handicap_by_oddsband.csv"

    summary.to_csv(summary_path, sep=";", decimal=",", index=False)
    by_band.to_csv(bands_path, sep=";", decimal=",", index=False)
    move_by_band.to_csv(move_path, sep=";", decimal=",", index=False)

    # print
    print("\n=== OP BACKTEST (BET @ OP, SETTLE @ OP) ===")
    print(summary.to_string(index=False))
    print("\n--- By odds band (OP) ---")
    print(by_band.to_string(index=False))

    print("\n=== DIAGNOSTIC: OP -> SP SHORTENING (NOT SETTLEMENT) ===")
    if len(move_by_band):
        print(move_by_band.to_string(index=False))
    else:
        print("(No SP available for diagnostics)")

    print("\nWrote:")
    print(f"- {summary_path}")
    print(f"- {bands_path}")
    print(f"- {move_path}")


if __name__ == "__main__":
    main()
