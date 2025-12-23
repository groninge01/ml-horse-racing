#!/usr/bin/env python3
"""
rolling_forward_elo_handicap_smallfield.py

Rolling forward validation of Elo edge using STRICT filters:

- Handicap races only
- Field size <= 14
- Elo-only signal
- OP only
- Flat staking
- Rolling time windows (18m train / 6m test)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd


# =============================
# Config
# =============================

INPUT_PATH = Path("output") / "horse_ml_scored_test_with_elo_and_type.csv"
OUTPUT_DIR = Path("output")

# Rolling window sizes (adaptive)
TRAIN_MONTHS = 18
TEST_MONTHS = 6

EDGE_QUANTILE = 0.90
STAKE = 1.0

MAX_OP_DEC = 12.0
MIN_OP_DEC = 1.01


# =============================
# EU helpers
# =============================

def read_csv_eu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df

def to_csv_eu(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        sep=";",
        decimal=",",
        date_format="%d-%m-%Y",
    )


# =============================
# Helpers
# =============================

def profit_flat(odds: float, y: int) -> float:
    return (odds - 1.0) if y == 1 else -1.0

def get_time_col(df: pd.DataFrame) -> pd.Series:
    if "runner_race_dt" in df.columns:
        t = pd.to_datetime(df["runner_race_dt"], errors="coerce", dayfirst=True)
        if t.notna().any():
            return t
    return pd.to_datetime(df["date"], errors="coerce", dayfirst=True)


# =============================
# Main
# =============================

def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Missing {INPUT_PATH}")
        return 2

    df = read_csv_eu(INPUT_PATH)

    required = ["race_name", "op_dec", "y", "p_elo", "ran"]
    for c in required:
        if c not in df.columns:
            print(f"Missing column: {c}")
            return 2

    df = df.copy()
    df["t"] = get_time_col(df)

    df["op_dec"] = pd.to_numeric(df["op_dec"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    df["p_elo"] = pd.to_numeric(df["p_elo"], errors="coerce")
    df["ran"] = pd.to_numeric(df["ran"], errors="coerce")

    df = df.dropna(subset=["t", "op_dec", "p_elo", "ran"])
    df = df[(df["op_dec"] >= MIN_OP_DEC) & (df["op_dec"] <= MAX_OP_DEC)]

    # Filters
    df = df[df["race_name"].str.contains("handicap", case=False, na=False)]
    df = df[df["ran"] <= 14]

    if df.empty:
        print("No rows after filtering.")
        return 0

    # Elo edge
    df["op_implied"] = 1.0 / df["op_dec"]
    df["edge_elo"] = df["p_elo"] - df["op_implied"]

    df = df.sort_values("t").reset_index(drop=True)

    start_date = df["t"].min()
    end_date = df["t"].max()

    window_results = []
    all_trades = []

    current_start = start_date

    while True:
        train_end = current_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        train = df[(df["t"] >= current_start) & (df["t"] < train_end)]
        test = df[(df["t"] >= train_end) & (df["t"] < test_end)]

        if len(train) < 200 or len(test) < 50:
            break

        cutoff = train["edge_elo"].quantile(EDGE_QUANTILE)

        test_bets = test[test["edge_elo"] >= cutoff].copy()
        test_bets["profit"] = test_bets.apply(
            lambda r: profit_flat(r["op_dec"], r["y"]), axis=1
        )

        profit = test_bets["profit"].sum()
        bets = len(test_bets)
        winners = int(test_bets["y"].sum())
        roi = profit / bets if bets > 0 else np.nan

        window_results.append({
            "train_start": current_start.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "train_rows": len(train),
            "test_rows": len(test),
            "bets": bets,
            "winners": winners,
            "profit": profit,
            "roi": roi,
            "edge_cutoff": cutoff,
        })

        if bets > 0:
            test_bets["train_start"] = current_start.date()
            test_bets["test_start"] = train_end.date()
            all_trades.append(test_bets)

        current_start = train_end

        if train_end >= end_date:
            break

    if not window_results:
        print("\nNo valid rolling windows could be formed with current filters.")
        print("This means the edge exists, but data density is limited.")
        return 0

    summary = pd.DataFrame(window_results)
    to_csv_eu(summary, OUTPUT_DIR / "rolling_elo_window_summary.csv")

    trades = pd.concat(all_trades, ignore_index=True)
    to_csv_eu(trades, OUTPUT_DIR / "rolling_elo_all_trades.csv")

    print("\n=== ROLLING FORWARD ELO TEST (HANDICAP, FIELD <=14) ===")
    print(f"Windows tested: {len(summary)}")
    print(f"Total bets: {summary['bets'].sum():,}")
    print(f"Aggregate ROI: {(summary['profit'].sum() / summary['bets'].sum()):.3f}")

    print("\nPer-window ROI:")
    print(summary[["train_start", "test_start", "bets", "roi"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
