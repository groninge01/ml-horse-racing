#!/usr/bin/env python3
"""
horse_elo.py

Elo-style rating for horse racing (multi-runner, ordinal outcomes).

Key idea:
- Each race is treated as many pairwise comparisons.
- Actual score for a horse = fraction of opponents it "beat" in that race.
- Expected score = average win probability vs each opponent based on current Elo ratings.
- Update Elo sequentially in time order.

Post-race fields (RPR/TS) are NOT used for prediction.
They can optionally be used to weight the Elo update AFTER the race (race informativeness).

INPUT:
- Your historical raceform.csv (comma-separated typical)
  Must contain at least: date, race_id, horse, pos
  Recommended: off, rpr, ts

OUTPUT (EU CSV) to ./output:
- output/horse_elo_runner_history.csv
- output/horse_elo_final_ratings.csv

EU formatting:
- sep=';'
- decimal=','
- date_format='DD-MM-YYYY'
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

OUTPUT_DIR = Path("output")

# Elo parameters
ELO_START = 1500.0
K_BASE = 32.0

# Elo scale (classic chess uses 400)
ELO_SCALE = 400.0

# How to treat non-numeric positions (PU/UR/F/etc.)
# We treat them as "worst finish" in the race (tied at bottom).
NONFINISH_AS_BOTTOM = True

# Race weighting
USE_RACE_WEIGHT = True

# Weight components (post-race, allowed for update only)
USE_RPR_WEIGHT = True
USE_TS_WEIGHT = False  # toggle if TS exists and you want it

# Clamp weights to avoid extreme influence
WEIGHT_MIN = 0.75
WEIGHT_MAX = 1.35

# Field-size scaling: larger fields = slightly more informative
FIELD_SIZE_EXPONENT = 0.50  # sqrt(field_size-1)

# Minimum runners to update (need at least 2)
MIN_RUNNERS = 2


# -----------------------------
# EU CSV helpers
# -----------------------------

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
# Text / parsing helpers
# -----------------------------

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Try comma-separated first; if it looks wrong, try EU semicolon.
    """
    # Try comma
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] >= 5:
            return df
    except Exception:
        pass

    # Try EU style
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    return df

def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'date' and 'off' into a sortable timestamp.
    If 'off' missing, use date only.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "off" in out.columns:
        # 'off' might be like '12:30' or '1:05'
        off = out["off"].astype(str).str.strip()
        # Build a datetime string; if off is invalid, fallback to midnight
        dt_str = out["date"].dt.strftime("%Y-%m-%d") + " " + off
        out["race_dt"] = pd.to_datetime(dt_str, errors="coerce")
        # Fill missing race_dt with date only
        out["race_dt"] = out["race_dt"].fillna(out["date"])
    else:
        out["race_dt"] = out["date"]

    return out


# -----------------------------
# Elo math
# -----------------------------

def elo_win_prob(rating_a: float, rating_b: float, scale: float = ELO_SCALE) -> float:
    """
    Elo expected score (probability A beats B).
    """
    # Classic: 1 / (1 + 10^((Rb-Ra)/400))
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))

def compute_expected_scores(ratings: np.ndarray) -> np.ndarray:
    """
    For N runners, expected score for i is average win prob vs all j != i.
    """
    n = len(ratings)
    if n <= 1:
        return np.zeros(n)

    exp = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            s += elo_win_prob(ratings[i], ratings[j])
        exp[i] = s / (n - 1)
    return exp

def compute_actual_scores_from_pos(pos: pd.Series) -> np.ndarray:
    """
    Actual score for horse i = fraction of opponents it "beat" (pairwise).
    Based on finishing position ordering.

    If positions are missing or non-numeric:
    - if NONFINISH_AS_BOTTOM=True, treat as tied at bottom
    - else drop those runners (we choose bottom-tie by default)
    """
    # Convert to numeric where possible
    pos_num = pd.to_numeric(pos, errors="coerce")

    n = len(pos_num)
    if n <= 1:
        return np.zeros(n)

    # Nonfinish handling
    if NONFINISH_AS_BOTTOM:
        # Non-numeric => big number (worst)
        worst = (np.nanmax(pos_num.values) if np.isfinite(np.nanmax(pos_num.values)) else n) + 1000
        pos_ord = pos_num.fillna(worst).astype(float).values
    else:
        # Drop NaNs by marking very worst as well (keeps array length stable)
        worst = (np.nanmax(pos_num.values) if np.isfinite(np.nanmax(pos_num.values)) else n) + 1000
        pos_ord = pos_num.fillna(worst).astype(float).values

    # Pairwise wins:
    # If pos_i < pos_j => i beat j
    # If equal => tie => 0.5
    actual = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            if pos_ord[i] < pos_ord[j]:
                s += 1.0
            elif pos_ord[i] == pos_ord[j]:
                s += 0.5
            else:
                s += 0.0
        actual[i] = s / (n - 1)
    return actual


def race_weight(group: pd.DataFrame) -> float:
    """
    Compute a multiplicative weight for the Elo update for this race.

    Uses:
    - field size scaling (sqrt(n-1))
    - optional RPR mean scaling (post-race, allowed for updating only)
    - optional TS mean scaling

    All clamped to [WEIGHT_MIN, WEIGHT_MAX] after combination (excluding field scaling).
    """
    n = len(group)
    if n < MIN_RUNNERS:
        return 0.0

    w = 1.0

    if USE_RPR_WEIGHT and "rpr" in group.columns:
        rpr = pd.to_numeric(group["rpr"], errors="coerce")
        m = float(np.nanmean(rpr.values)) if np.isfinite(np.nanmean(rpr.values)) else np.nan
        if np.isfinite(m):
            # Mild scaling around ~90-110 typical bands; keep gentle.
            # Example: mean 100 => ~1.00, mean 90 => 0.95, mean 110 => 1.05
            w *= np.clip(1.0 + (m - 100.0) / 200.0, 0.90, 1.10)

    if USE_TS_WEIGHT and "ts" in group.columns:
        ts = pd.to_numeric(group["ts"], errors="coerce")
        m = float(np.nanmean(ts.values)) if np.isfinite(np.nanmean(ts.values)) else np.nan
        if np.isfinite(m):
            w *= np.clip(1.0 + (m - 100.0) / 250.0, 0.90, 1.10)

    # Clamp the non-field part
    w = float(np.clip(w, WEIGHT_MIN, WEIGHT_MAX))

    # Field size scaling
    field_scale = float((max(n - 1, 1)) ** FIELD_SIZE_EXPONENT)

    return w * field_scale


# -----------------------------
# Main
# -----------------------------

def main(csv_path: str) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = robust_read_csv(csv_path)
    df.columns = [normalize_text(c) for c in df.columns]

    required = ["date", "race_id", "horse", "pos"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Found columns: {list(df.columns)[:60]}")
        return 2

    # Basic normalize
    df["horse"] = df["horse"].astype(str).map(normalize_text)
    df["race_id"] = df["race_id"].astype(str).map(normalize_text)

    # Parse timestamps
    df = parse_datetime_columns(df)
    df = df.dropna(subset=["race_dt"])

    # Sort chronologically (strict)
    df = df.sort_values(["race_dt", "race_id"]).reset_index(drop=True)

    # Prepare rating store
    ratings: Dict[str, float] = {}

    rows_out = []

    # Group by race in time order
    for race_id, g in df.groupby("race_id", sort=False):
        g = g.copy()
        n = len(g)
        if n < MIN_RUNNERS:
            continue

        # Current ratings for each horse (pre-race)
        horses = g["horse"].tolist()
        pre = np.array([ratings.get(h, ELO_START) for h in horses], dtype=float)

        # Compute expected & actual
        expected = compute_expected_scores(pre)
        actual = compute_actual_scores_from_pos(g["pos"])

        # Race K scaling
        rw = race_weight(g) if USE_RACE_WEIGHT else float((max(n - 1, 1)) ** FIELD_SIZE_EXPONENT)
        k = K_BASE * rw

        delta = k * (actual - expected)
        post = pre + delta

        # Write back to rating store
        for h, new_r in zip(horses, post):
            ratings[h] = float(new_r)

        # Store per-runner history row
        # Include useful fields if present
        for idx, (h, pre_r, post_r, exp_s, act_s, d) in enumerate(zip(horses, pre, post, expected, actual, delta)):
            row = {
                "date": g.iloc[idx]["date"] if "date" in g.columns else pd.NaT,
                "race_dt": g.iloc[idx]["race_dt"],
                "race_id": race_id,
                "horse": h,
                "pos": g.iloc[idx]["pos"],
                "field_size": n,
                "pre_elo": float(pre_r),
                "expected_score": float(exp_s),
                "actual_score": float(act_s),
                "delta_elo": float(d),
                "post_elo": float(post_r),
                "k_used": float(k),
            }
            for col in ["course", "off", "race_name", "type", "class", "dist", "going", "ran", "draw", "age", "sex", "wgt", "or", "rpr", "ts"]:
                if col in g.columns:
                    row[col] = g.iloc[idx][col]
            rows_out.append(row)

    hist = pd.DataFrame(rows_out)

    if hist.empty:
        print("No races processed (check your file contents).")
        return 0

    # Output per-runner history (EU CSV)
    # Drop race_dt or keep it; it can be useful, but it includes time.
    # We'll keep it as ISO string for clarity.
    hist_out = hist.copy()
    hist_out["race_dt"] = pd.to_datetime(hist_out["race_dt"], errors="coerce")
    # Keep date in date column, and race_dt as string to avoid time formatting confusion in EU CSV
    hist_out["race_dt"] = hist_out["race_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    to_csv_eu(hist_out, OUTPUT_DIR / "horse_elo_runner_history.csv")
    print("Wrote: output/horse_elo_runner_history.csv")

    # Final ratings table
    final = (
        pd.DataFrame({"horse": list(ratings.keys()), "elo": list(ratings.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    to_csv_eu(final, OUTPUT_DIR / "horse_elo_final_ratings.csv")
    print("Wrote: output/horse_elo_final_ratings.csv")

    # Console quick stats
    print("\n=== Elo build summary ===")
    print(f"Processed runner rows: {len(hist_out):,}")
    print(f"Unique horses rated:   {len(final):,}")
    print(f"Elo start:            {ELO_START}")
    print(f"K base:               {K_BASE} (scaled by race weight)")
    print(f"OP/SP not used. RPR/TS used only for race weighting: RPR={USE_RPR_WEIGHT}, TS={USE_TS_WEIGHT}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python horse_elo.py data/raceform.csv")
    sys.exit(main(sys.argv[1]))
