#!/usr/bin/env python3
"""
elo_segment_analysis.py

Forward-only analysis:
Where does the Elo edge work best?

Segments:
- race type (from authoritative `type` column)
- handicap vs non-handicap (from `race_name`)
- field size buckets
- type × handicap interaction

ASSUMPTIONS:
- `type` exists (Hurdle, Chase, Flat, etc.)
- handicap inferred from 'handicap' in race_name
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd


# =============================
# Config (must match forward test)
# =============================

INPUT_PATH = Path("output") / "horse_ml_scored_test_with_elo_and_type.csv"
OUTPUT_DIR = Path("output")

MAX_OP_DEC = 12.0
MIN_OP_DEC = 1.01

CALIBRATION_FRAC = 0.70
EDGE_QUANTILE = 0.90
STAKE = 1.0


# =============================
# EU CSV helpers
# =============================

def read_csv_eu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df

def to_csv_eu(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", decimal=",", date_format="%d-%m-%Y")


# =============================
# Helpers
# =============================

def profit_flat(odds: float, y: int) -> float:
    return (odds - 1.0) if y == 1 else -1.0


def summarize(df: pd.DataFrame, group_cols):
    """
    If group_cols is empty -> overall summary row.
    Otherwise -> grouped summary.
    """
    df = df.copy()
    df["profit"] = df.apply(lambda r: profit_flat(r["op_dec"], r["y"]), axis=1)

    if not group_cols:
        bets = len(df)
        winners = int(df["y"].sum())
        profit = float(df["profit"].sum())
        return pd.DataFrame([{
            "bets": bets,
            "winners": winners,
            "strike": winners / bets if bets > 0 else np.nan,
            "avg_odds": df["op_dec"].mean(),
            "profit": profit,
            "roi": profit / bets if bets > 0 else np.nan,
        }])

    g = df.groupby(group_cols, dropna=False)

    out = g.agg(
        bets=("y", "size"),
        winners=("y", "sum"),
        strike=("y", "mean"),
        avg_odds=("op_dec", "mean"),
        profit=("profit", "sum"),
    ).reset_index()

    out["roi"] = out["profit"] / out["bets"]
    return out.sort_values("roi", ascending=False)


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

    required = ["op_dec", "y", "p_elo", "type", "race_name"]
    for c in required:
        if c not in df.columns:
            print(f"ERROR: missing required column '{c}'")
            print("Available columns:")
            print(list(df.columns))
            return 2

    df = df.copy()
    df["t"] = get_time_col(df)

    df["op_dec"] = pd.to_numeric(df["op_dec"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    df["p_elo"] = pd.to_numeric(df["p_elo"], errors="coerce")

    df = df.dropna(subset=["t", "op_dec", "p_elo"])
    df = df[(df["op_dec"] >= MIN_OP_DEC) & (df["op_dec"] <= MAX_OP_DEC)]

    # Build Elo edge
    df["op_implied"] = 1.0 / df["op_dec"]
    df["edge_elo"] = df["p_elo"] - df["op_implied"]

    # Chronological split
    df = df.sort_values("t").reset_index(drop=True)
    split_idx = int(len(df) * CALIBRATION_FRAC)

    calib = df.iloc[:split_idx]
    fwd = df.iloc[split_idx:]

    cutoff = calib["edge_elo"].quantile(EDGE_QUANTILE)
    bets = fwd[fwd["edge_elo"] >= cutoff].copy()

    if bets.empty:
        print("No forward bets at this cutoff.")
        return 0

    # -----------------------------
    # Segment construction
    # -----------------------------

    # Race type (authoritative)
    bets["race_type"] = bets["type"].astype(str).str.lower()

    # Handicap (from race_name)
    bets["handicap"] = bets["race_name"].astype(str).str.contains(
        "handicap", case=False, na=False
    )

    # Field size buckets
    if "ran" in bets.columns:
        bets["field_bucket"] = pd.cut(
            bets["ran"],
            bins=[0, 7, 10, 14, 20, 99],
            labels=["<=7", "8-10", "11-14", "15-20", "21+"],
        )
    else:
        bets["field_bucket"] = "unknown"

    # -----------------------------
    # Analyses
    # -----------------------------

    to_csv_eu(
        summarize(bets, []),
        OUTPUT_DIR / "elo_segment_summary.csv",
    )

    to_csv_eu(
        summarize(bets, ["race_type"]),
        OUTPUT_DIR / "elo_by_race_type.csv",
    )

    to_csv_eu(
        summarize(bets, ["handicap"]),
        OUTPUT_DIR / "elo_by_handicap.csv",
    )

    to_csv_eu(
        summarize(bets, ["field_bucket"]),
        OUTPUT_DIR / "elo_by_field_size.csv",
    )

    to_csv_eu(
        summarize(bets, ["race_type", "handicap"]),
        OUTPUT_DIR / "elo_by_type_and_handicap.csv",
    )

    print("\n=== ELO SEGMENT ANALYSIS (FORWARD ONLY) ===")
    print(f"Forward bets analysed: {len(bets):,}")
    print(f"Elo edge cutoff used:  {cutoff:.6f}")
    print("\nTop segments (type × handicap):")
    print(
        summarize(bets, ["race_type", "handicap"])
        .head(10)
        .to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
