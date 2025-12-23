#!/usr/bin/env python3
"""
merge_elo_and_rerun_op_analysis.py

FINAL FIX:
- merge_asof requires sorting by [time_key, by_key] — in that order.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# =============================
# Config
# =============================

OUTPUT_DIR = Path("output")
SCORED_PATH = OUTPUT_DIR / "horse_ml_scored_test.csv"
ELO_HIST_PATH = OUTPUT_DIR / "horse_elo_runner_history.csv"

MAX_OP_DEC = 12.0
STAKE = 1.0
TOP_N = 200
ELO_SCALE = 400.0


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
# Time helpers
# =============================

def build_runner_race_dt(df: pd.DataFrame) -> pd.Series:
    d = df["date"]
    if "off" in df.columns:
        off = df["off"].astype(str).str.strip()
        dt = pd.to_datetime(d.dt.strftime("%Y-%m-%d") + " " + off, errors="coerce")
        return dt.fillna(d)
    return d


# =============================
# Elo helpers
# =============================

def elo_implied_prob(elo: pd.Series) -> pd.Series:
    x = (elo - 1500.0) / ELO_SCALE
    return 1.0 / (1.0 + np.power(10.0, -x))


# =============================
# Betting helpers
# =============================

def profit_from_odds_and_result(odds: pd.Series, y: pd.Series) -> pd.Series:
    return np.where(y == 1, odds - 1.0, -1.0)

def odds_band(op_dec: pd.Series) -> pd.Categorical:
    bins = [0, 2, 3, 5, 8, 12, np.inf]
    labels = ["<=2.0", "2-3", "3-5", "5-8", "8-12", ">12"]
    return pd.cut(op_dec, bins=bins, labels=labels, include_lowest=True)


# =============================
# Analysis helpers
# =============================

def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["profit"] = profit_from_odds_and_result(df["op_dec"], df["y"])
    g = df.groupby(group_cols, observed=False)

    out = g.agg(
        runners=("y", "size"),
        winners=("y", "sum"),
        win_rate=("y", "mean"),
        avg_odds=("op_dec", "mean"),
        avg_edge=("edge", "mean"),
        profit=("profit", "sum"),
    ).reset_index()

    out["roi"] = out["profit"] / out["runners"]
    return out


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal_rank"] = df.groupby("race_id")["signal_p"].rank(
        method="min", ascending=False
    )
    df["op_rank"] = df.groupby("race_id")["op_dec"].rank(
        method="min", ascending=True
    )
    df["rank_gap"] = df["op_rank"] - df["signal_rank"]
    return df


def run_analysis(df: pd.DataFrame, name: str) -> None:
    df = df.copy()
    df["edge"] = df["signal_p"] - df["op_implied"]
    df = compute_ranks(df)

    to_csv_eu(
        summarize(df, ["signal_rank", "odds_band"]),
        OUTPUT_DIR / f"op_{name}_by_rank_and_oddsband.csv",
    )

    df["edge_q"] = pd.qcut(df["edge"], 10, duplicates="drop")
    to_csv_eu(
        summarize(df, ["edge_q"]).sort_values("roi", ascending=False),
        OUTPUT_DIR / f"op_{name}_by_edge_quantile.csv",
    )

    df["rank_gap_bucket"] = pd.cut(
        df["rank_gap"],
        bins=[-999, -5, -3, -1, 0, 2, 4, 999],
        labels=[
            "market>>>signal",
            "market>>signal",
            "market>signal",
            "agree",
            "signal>market",
            "signal>>market",
            "signal>>>market",
        ],
    )

    to_csv_eu(
        summarize(df, ["rank_gap_bucket", "odds_band"]),
        OUTPUT_DIR / f"op_{name}_by_rank_gap.csv",
    )

    to_csv_eu(
        df.sort_values("edge", ascending=False).head(TOP_N),
        OUTPUT_DIR / f"op_{name}_top_candidates.csv",
    )


# =============================
# Main
# =============================

def main() -> int:
    scored = read_csv_eu(SCORED_PATH)
    elo = read_csv_eu(ELO_HIST_PATH)

    # ---- clean scored
    scored["race_id"] = scored["race_id"].astype(str)
    scored["horse"] = scored["horse"].astype(str)
    scored["y"] = scored["y"].fillna(0).astype(int)
    scored["op_dec"] = pd.to_numeric(scored["op_dec"], errors="coerce")
    scored["p_win_form"] = pd.to_numeric(scored["p_win_form"], errors="coerce")

    scored["runner_race_dt"] = build_runner_race_dt(scored)
    scored = scored.dropna(subset=["runner_race_dt", "op_dec", "p_win_form"])
    scored = scored[(scored["op_dec"] > 1.0) & (scored["op_dec"] <= MAX_OP_DEC)]

    # ---- clean elo
    elo["horse"] = elo["horse"].astype(str)
    elo["race_dt"] = pd.to_datetime(elo["race_dt"], errors="coerce")
    elo["post_elo"] = pd.to_numeric(elo["post_elo"], errors="coerce")
    elo = elo.dropna(subset=["race_dt", "post_elo"])

    # 🔑 CRITICAL SORT (time first, then horse)
    scored = scored.sort_values(["runner_race_dt", "horse"]).reset_index(drop=True)
    elo = elo.sort_values(["race_dt", "horse"]).reset_index(drop=True)

    # ---- merge as-of
    merged = pd.merge_asof(
        scored,
        elo[["horse", "race_dt", "post_elo"]],
        left_on="runner_race_dt",
        right_on="race_dt",
        by="horse",
        direction="backward",
        allow_exact_matches=True,
    )

    merged["elo_pre"] = merged["post_elo"].fillna(1500.0)
    merged["p_elo"] = elo_implied_prob(merged["elo_pre"])
    merged["p_ens"] = (merged["p_win_form"] + merged["p_elo"]) / 2.0

    merged["op_implied"] = 1.0 / merged["op_dec"]
    merged["odds_band"] = odds_band(merged["op_dec"])

    to_csv_eu(merged, OUTPUT_DIR / "horse_ml_scored_test_with_elo.csv")

    # ---- analyses
    for name, col in [
        ("form", "p_win_form"),
        ("elo", "p_elo"),
        ("ens", "p_ens"),
    ]:
        df = merged.copy()
        df["signal_p"] = df[col]
        run_analysis(df, name)

    print("\n=== MERGE + OP ANALYSIS COMPLETE ===")
    print(f"Rows analysed: {len(merged):,}")
    print(f"Missing Elo filled with 1500: {(merged['post_elo'].isna()).sum():,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
