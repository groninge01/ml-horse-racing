#!/usr/bin/env python3
"""
forward_test_elo_edge.py

Strict forward validation of "top Elo edge" betting rule.

Input (EU CSV):
- output/horse_ml_scored_test_with_elo.csv

Required columns:
- date (DD-MM-YYYY) OR runner_race_dt (datetime-like)
- race_id
- horse
- op_dec
- y (1 win, 0 otherwise)
- p_elo (Elo-derived strength proxy in (0,1))

Rule:
- Compute op_implied = 1/op_dec
- edge_elo = p_elo - op_implied
- Determine cutoff = quantile(edge_elo, q=0.90) on CALIBRATION period only
- Bet in FORWARD period if edge_elo >= cutoff

Outputs (EU CSVs to ./output):
- output/forward_elo_summary.csv
- output/forward_elo_trades.csv
- output/forward_elo_equity_curve.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

INPUT_PATH = Path("output") / "horse_ml_scored_test_with_elo.csv"
OUTPUT_DIR = Path("output")

# Odds cap (keep consistent with earlier work)
MAX_OP_DEC = 12.0
MIN_OP_DEC = 1.01

# Train/forward split by time (fraction in calibration)
CALIBRATION_FRAC = 0.70

# "Top edge" definition (10% = top decile)
EDGE_QUANTILE = 0.90

# Flat stake per bet
STAKE = 1.0


# -----------------------------
# EU CSV helpers
# -----------------------------

def read_csv_eu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    # parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df

def to_csv_eu(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", decimal=",", date_format="%d-%m-%Y")


# -----------------------------
# Core
# -----------------------------

def profit_flat(odds_dec: float, y: int, stake: float) -> float:
    if y == 1:
        return (odds_dec - 1.0) * stake
    return -stake

def compute_drawdown(equity: np.ndarray) -> Tuple[float, int]:
    """
    Returns (max_drawdown, max_dd_duration_bets)
    max_drawdown is negative number (e.g. -25.0)
    duration is length in bets from peak to recovery (or end if never recovered)
    """
    peak = -np.inf
    peak_idx = 0
    max_dd = 0.0
    max_dur = 0
    for i, v in enumerate(equity):
        if v > peak:
            peak = v
            peak_idx = i
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
            # duration so far (until current i)
            max_dur = i - peak_idx
    return float(max_dd), int(max_dur)

def max_losing_streak(y: np.ndarray) -> int:
    """
    y is 1 for win, 0 for loss.
    """
    streak = 0
    best = 0
    for v in y:
        if v == 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)

def get_time_col(df: pd.DataFrame) -> pd.Series:
    """
    Prefer runner_race_dt if available (contains time),
    else fall back to date.
    """
    if "runner_race_dt" in df.columns:
        t = pd.to_datetime(df["runner_race_dt"], errors="coerce")
        if t.notna().any():
            return t
    if "date" in df.columns:
        return df["date"]
    raise ValueError("Need 'runner_race_dt' or 'date' column.")

def summarize_period(name: str, bets: pd.DataFrame) -> Dict[str, Any]:
    if bets.empty:
        return {
            "period": name,
            "bets": 0,
            "winners": 0,
            "strike": np.nan,
            "profit": 0.0,
            "roi": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "max_losing_streak": np.nan,
            "max_drawdown": np.nan,
            "max_dd_duration_bets": np.nan,
            "from": "",
            "to": "",
        }

    profit = bets["profit"].sum()
    roi = profit / (len(bets) * STAKE)
    strike = bets["y"].mean()
    avg_odds = bets["op_dec"].mean()
    avg_edge = bets["edge_elo"].mean()

    equity = bets["equity"].values
    max_dd, max_dur = compute_drawdown(equity)
    mls = max_losing_streak(bets["y"].values)

    return {
        "period": name,
        "bets": int(len(bets)),
        "winners": int(bets["y"].sum()),
        "strike": float(strike),
        "profit": float(profit),
        "roi": float(roi),
        "avg_odds": float(avg_odds),
        "avg_edge": float(avg_edge),
        "max_losing_streak": int(mls),
        "max_drawdown": float(max_dd),
        "max_dd_duration_bets": int(max_dur),
        "from": str(bets["t"].min()),
        "to": str(bets["t"].max()),
    }


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: missing {INPUT_PATH}")
        return 2

    df = read_csv_eu(INPUT_PATH)

    required = ["race_id", "horse", "op_dec", "y", "p_elo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing required columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        return 2

    df = df.copy()
    df["t"] = get_time_col(df)

    df["race_id"] = df["race_id"].astype(str)
    df["horse"] = df["horse"].astype(str)
    df["op_dec"] = pd.to_numeric(df["op_dec"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    df["p_elo"] = pd.to_numeric(df["p_elo"], errors="coerce")

    # Clean filters
    df = df.dropna(subset=["t", "op_dec", "p_elo"])
    df = df[(df["op_dec"] >= MIN_OP_DEC) & (df["op_dec"] <= MAX_OP_DEC)]
    df = df[(df["p_elo"] > 0) & (df["p_elo"] < 1)]

    if df.empty:
        print("No rows after filtering.")
        return 0

    # Build edge
    df["op_implied"] = 1.0 / df["op_dec"].astype(float)
    df["edge_elo"] = df["p_elo"].astype(float) - df["op_implied"].astype(float)

    # Sort chronologically
    df = df.sort_values(["t", "race_id", "horse"]).reset_index(drop=True)

    # Split by time index (not random)
    split_idx = int(len(df) * CALIBRATION_FRAC)
    calib = df.iloc[:split_idx].copy()
    fwd = df.iloc[split_idx:].copy()

    # Cutoff from calibration only
    cutoff = float(calib["edge_elo"].quantile(EDGE_QUANTILE))

    # Apply rule
    calib_bets = calib[calib["edge_elo"] >= cutoff].copy()
    fwd_bets = fwd[fwd["edge_elo"] >= cutoff].copy()

    # Compute profits & equity in each period (independent)
    def add_profit_equity(bets: pd.DataFrame) -> pd.DataFrame:
        if bets.empty:
            return bets
        bets = bets.sort_values(["t", "race_id", "horse"]).reset_index(drop=True)
        bets["profit"] = bets.apply(lambda r: profit_flat(float(r["op_dec"]), int(r["y"]), STAKE), axis=1)
        bets["equity"] = bets["profit"].cumsum()
        return bets

    calib_bets = add_profit_equity(calib_bets)
    fwd_bets = add_profit_equity(fwd_bets)

    # Outputs
    # Trades
    trade_cols_prefer = [
        "t", "date", "course", "off", "race_name", "race_id", "horse",
        "op_dec", "op_implied", "p_elo", "edge_elo",
        "y", "profit", "equity",
        "elo_pre",  # if present
    ]
    trade_cols = [c for c in trade_cols_prefer if c in fwd_bets.columns]
    trades_out = fwd_bets[trade_cols].copy()
    to_csv_eu(trades_out, OUTPUT_DIR / "forward_elo_trades.csv")

    # Equity curve
    eq = fwd_bets[["t", "equity"]].copy() if not fwd_bets.empty else pd.DataFrame(columns=["t", "equity"])
    to_csv_eu(eq, OUTPUT_DIR / "forward_elo_equity_curve.csv")

    # Summary
    summary_rows = []
    summary_rows.append({
        "setting": "CALIBRATION_FRAC",
        "value": CALIBRATION_FRAC
    })
    summary_rows.append({
        "setting": "EDGE_QUANTILE",
        "value": EDGE_QUANTILE
    })
    summary_rows.append({
        "setting": "EDGE_CUTOFF_FROM_CALIB",
        "value": cutoff
    })
    summary_rows.append({
        "setting": "ODDS_RANGE",
        "value": f"{MIN_OP_DEC}..{MAX_OP_DEC}"
    })

    summary_df_settings = pd.DataFrame(summary_rows)

    perf = pd.DataFrame([
        summarize_period("calibration (bets only)", calib_bets),
        summarize_period("forward (bets only)", fwd_bets),
    ])

    # Write summaries
    to_csv_eu(summary_df_settings, OUTPUT_DIR / "forward_elo_settings.csv")
    to_csv_eu(perf, OUTPUT_DIR / "forward_elo_summary.csv")

    # Console
    print("\n=== FORWARD TEST (ELO EDGE) ===")
    print(f"Rows total:         {len(df):,}")
    print(f"Calibration rows:   {len(calib):,}   | bets: {len(calib_bets):,}")
    print(f"Forward rows:       {len(fwd):,}   | bets: {len(fwd_bets):,}")
    print(f"Edge cutoff (q={EDGE_QUANTILE:.2f} from calibration): {cutoff:.6f}")
    if len(fwd_bets) > 0:
        print(f"Forward ROI:        {perf.loc[perf['period']=='forward (bets only)', 'roi'].iloc[0]:.3f}")
        print(f"Forward strike:     {perf.loc[perf['period']=='forward (bets only)', 'strike'].iloc[0]:.3f}")
        print(f"Forward max DD:     {perf.loc[perf['period']=='forward (bets only)', 'max_drawdown'].iloc[0]:.2f}")
        print(f"Forward max L-stk:  {perf.loc[perf['period']=='forward (bets only)', 'max_losing_streak'].iloc[0]}")
    else:
        print("No forward bets met the cutoff (try lowering EDGE_QUANTILE).")

    print("\nWrote:")
    print(" - output/forward_elo_settings.csv")
    print(" - output/forward_elo_summary.csv")
    print(" - output/forward_elo_trades.csv")
    print(" - output/forward_elo_equity_curve.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
