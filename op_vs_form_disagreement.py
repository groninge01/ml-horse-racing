#!/usr/bin/env python3
"""
op_vs_form_disagreement.py

Analyze disagreement between a pure-form model and Opening Price (OP).

INPUT:
- output/horse_ml_scored_test.csv  (EU formatted: sep=';', decimal=',')

REQUIRES COLUMNS (from your pipeline):
- race_id
- date
- horse (optional but helpful)
- y (1 if won else 0)
- p_win_form (model probability)
- op_dec (opening odds decimal)
- op_frac (optional)
- course/off/race_name (optional)

OUTPUT (EU CSVs to ./output):
1) output/op_form_by_rank_and_oddsband.csv
2) output/op_form_by_edge_quantile.csv
3) output/op_form_by_rank_gap.csv
4) output/op_form_top_candidates.csv

All CSV outputs:
- sep=';'
- decimal=','
- date format DD-MM-YYYY

Notes:
- This script is OP-only (does not use SP).
- Uses a flat stake of 1 unit for ROI simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

INPUT_PATH = Path("output") / "horse_ml_scored_test.csv"
OUTPUT_DIR = Path("output")

# Tradable odds cap to avoid longshot bias artifacts
MAX_OP_DEC = 12.0

# Optional minimum odds (set None to disable)
MIN_OP_DEC: Optional[float] = None

# Flat stake per bet
STAKE = 1.0

# Number of rows to export in "top candidates"
TOP_N = 200


# -----------------------------
# EU CSV helpers
# -----------------------------

def read_csv_eu(path: Path) -> pd.DataFrame:
    # Your files are EU formatted: sep=';', decimal=','
    # date column is DD-MM-YYYY (exported by our script)
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)

    # Parse date if present
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


# -----------------------------
# Core computations
# -----------------------------

def add_ranks_and_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - op_implied = 1/op_dec
    - edge = p_win_form - op_implied
    - form_rank: rank within race by p_win_form desc (best=1)
    - op_rank: rank within race by op_dec asc (fav=1)
    - rank_gap = op_rank - form_rank (positive => model likes it more than OP market)
    """
    out = df.copy()

    out["op_implied"] = 1.0 / out["op_dec"].astype(float)
    out["edge"] = out["p_win_form"].astype(float) - out["op_implied"]

    # Ranks (handle ties consistently)
    out["form_rank"] = (
        out.groupby("race_id")["p_win_form"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["op_rank"] = (
        out.groupby("race_id")["op_dec"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    out["rank_gap"] = out["op_rank"] - out["form_rank"]

    return out


def profit_from_odds_and_result(odds_dec: pd.Series, y: pd.Series, stake: float) -> pd.Series:
    """
    Flat stake profit:
      win: (odds_dec - 1)*stake
      lose: -stake
    """
    odds = odds_dec.astype(float)
    win = y.astype(int) == 1
    return np.where(win, (odds - 1.0) * stake, -stake)


def odds_band(op_dec: pd.Series) -> pd.Categorical:
    """
    Common racing-ish odds bands (decimal):
    <=2.0 (~evens)
    2-3
    3-5
    5-8
    8-12
    >12 (should be filtered out by MAX_OP_DEC)
    """
    bins = [0.0, 2.0, 3.0, 5.0, 8.0, 12.0, np.inf]
    labels = ["<=2.0", "2-3", "3-5", "5-8", "8-12", ">12"]
    return pd.cut(op_dec.astype(float), bins=bins, labels=labels, include_lowest=True, right=True)


def summarize_group(df: pd.DataFrame, group_cols: List[str], stake: float) -> pd.DataFrame:
    """
    Summarize win-rate, implied rate, avg odds, edge stats, ROI.
    """
    tmp = df.copy()
    tmp["profit"] = profit_from_odds_and_result(tmp["op_dec"], tmp["y"], stake)

    g = tmp.groupby(group_cols, dropna=False)

    out = g.agg(
        runners=("y", "size"),
        winners=("y", "sum"),
        win_rate=("y", "mean"),
        avg_op_dec=("op_dec", "mean"),
        avg_op_implied=("op_implied", "mean"),
        avg_p_win_form=("p_win_form", "mean"),
        avg_edge=("edge", "mean"),
        median_edge=("edge", "median"),
        profit=("profit", "sum"),
    ).reset_index()

    out["roi"] = out["profit"] / (out["runners"] * stake)
    return out.sort_values(group_cols)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("Run your training script first so it creates output/horse_ml_scored_test.csv")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_eu(INPUT_PATH)

    required = ["race_id", "y", "p_win_form", "op_dec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns in {INPUT_PATH}: {missing}")
        print(f"Found columns: {list(df.columns)}")
        return 2

    # Clean types
    df = df.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    df["p_win_form"] = pd.to_numeric(df["p_win_form"], errors="coerce")
    df["op_dec"] = pd.to_numeric(df["op_dec"], errors="coerce")

    # Filter: must have OP + model prob
    df = df.dropna(subset=["p_win_form", "op_dec"])

    # Odds caps
    df = df[df["op_dec"] > 1.0]
    df = df[df["op_dec"] <= MAX_OP_DEC]
    if MIN_OP_DEC is not None:
        df = df[df["op_dec"] >= float(MIN_OP_DEC)]

    if df.empty:
        print("No rows left after filtering (check OP availability and MAX_OP_DEC).")
        return 0

    # Add features for analysis
    df = add_ranks_and_edges(df)
    df["odds_band"] = odds_band(df["op_dec"])

    # -----------------------------
    # 1) Summary by form_rank × odds_band
    # -----------------------------
    by_rank_band = summarize_group(df, ["form_rank", "odds_band"], stake=STAKE)
    to_csv_eu(by_rank_band, OUTPUT_DIR / "op_form_by_rank_and_oddsband.csv")
    print("Wrote: output/op_form_by_rank_and_oddsband.csv")

    # -----------------------------
    # 2) Summary by edge quantile
    # -----------------------------
    # Quantiles across all runners (OP filtered)
    # 10 bins is usually a good start
    df["edge_q"] = pd.qcut(df["edge"], q=10, duplicates="drop")
    by_edge_q = summarize_group(df, ["edge_q"], stake=STAKE)

    # Add quantile boundaries for readability
    q_edges = df.groupby("edge_q")["edge"].agg(edge_min="min", edge_max="max").reset_index()
    by_edge_q = by_edge_q.merge(q_edges, on="edge_q", how="left")
    to_csv_eu(by_edge_q, OUTPUT_DIR / "op_form_by_edge_quantile.csv")
    print("Wrote: output/op_form_by_edge_quantile.csv")

    # -----------------------------
    # 3) Summary by rank_gap buckets
    # -----------------------------
    # rank_gap = op_rank - form_rank
    # Positive => model ranks horse higher than market does.
    def gap_bucket(x: int) -> str:
        if x >= 5:
            return "model>>market (>=+5)"
        if x >= 3:
            return "model>market (+3..+4)"
        if x >= 1:
            return "model>market (+1..+2)"
        if x == 0:
            return "agree (0)"
        if x >= -2:
            return "market>model (-1..-2)"
        if x >= -4:
            return "market>>model (-3..-4)"
        return "market>>>model (<=-5)"

    df["rank_gap_bucket"] = df["rank_gap"].apply(gap_bucket)

    by_gap = summarize_group(df, ["rank_gap_bucket", "odds_band"], stake=STAKE)
    # Order buckets nicely
    bucket_order = [
        "model>>market (>=+5)",
        "model>market (+3..+4)",
        "model>market (+1..+2)",
        "agree (0)",
        "market>model (-1..-2)",
        "market>>model (-3..-4)",
        "market>>>model (<=-5)",
    ]
    by_gap["rank_gap_bucket"] = pd.Categorical(by_gap["rank_gap_bucket"], categories=bucket_order, ordered=True)
    by_gap = by_gap.sort_values(["rank_gap_bucket", "odds_band"])
    to_csv_eu(by_gap, OUTPUT_DIR / "op_form_by_rank_gap.csv")
    print("Wrote: output/op_form_by_rank_gap.csv")

    # -----------------------------
    # 4) Top candidates list (for inspection)
    # -----------------------------
    # Highest edge first (model likes more than OP implied)
    cols_prefer = [
        "date", "course", "off", "race_name", "race_id", "horse",
        "op_frac", "op_dec", "op_implied",
        "p_win_form", "edge",
        "form_rank", "op_rank", "rank_gap",
        "ran", "draw", "age", "sex", "wgt_lbs", "or",
        "y",
    ]
    cols = [c for c in cols_prefer if c in df.columns]

    top = df.sort_values(["edge", "p_win_form"], ascending=[False, False]).head(TOP_N)[cols].copy()
    to_csv_eu(top, OUTPUT_DIR / "op_form_top_candidates.csv")
    print("Wrote: output/op_form_top_candidates.csv")

    # -----------------------------
    # Console quick summary
    # -----------------------------
    overall_profit = profit_from_odds_and_result(df["op_dec"], df["y"], STAKE).sum()
    overall_roi = overall_profit / (len(df) * STAKE)

    print("\n=== OP vs Form Disagreement (filtered) ===")
    print(f"Rows analyzed: {len(df):,}")
    print(f"OP cap: <= {MAX_OP_DEC}")
    if MIN_OP_DEC is not None:
        print(f"OP min: >= {MIN_OP_DEC}")
    print(f"Overall ROI (flat stake, betting ALL rows): {overall_roi:0.3f} (not a strategy, just a sanity check)")

    # Show a few best-performing edge quantiles
    best_edge = by_edge_q.sort_values("roi", ascending=False).head(5)
    print("\nTop 5 edge quantiles by ROI (diagnostic):")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 140):
        print(best_edge.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
