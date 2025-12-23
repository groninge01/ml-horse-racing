#!/usr/bin/env python3
"""
restore_type_and_merge.py

Re-attach authoritative race `type` (Hurdle, Chase, Flat, etc.)
from the original raceform.csv back into the Elo-scored dataset.

Input:
- data/raceform.csv
- output/horse_ml_scored_test_with_elo.csv

Output:
- output/horse_ml_scored_test_with_elo_and_type.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


# =============================
# Paths
# =============================

RACEFORM_PATH = Path("data") / "raceform.csv"
SCORED_PATH = Path("output") / "horse_ml_scored_test_with_elo.csv"
OUTPUT_PATH = Path("output") / "horse_ml_scored_test_with_elo_and_type.csv"


# =============================
# EU CSV helpers
# =============================

def read_csv_any(path: Path) -> pd.DataFrame:
    """
    raceform.csv is comma-separated, mixed encoding
    """
    return pd.read_csv(path, low_memory=False)

def read_csv_eu(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", low_memory=False)

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
# Main
# =============================

def main() -> int:
    if not RACEFORM_PATH.exists():
        print(f"ERROR: missing {RACEFORM_PATH}")
        return 2
    if not SCORED_PATH.exists():
        print(f"ERROR: missing {SCORED_PATH}")
        return 2

    print("Loading raceform.csv …")
    raceform = read_csv_any(RACEFORM_PATH)

    if "race_id" not in raceform.columns or "type" not in raceform.columns:
        print("ERROR: raceform.csv must contain 'race_id' and 'type'")
        print("Columns found:", list(raceform.columns))
        return 2

    # Keep only authoritative mapping
    race_types = (
        raceform[["race_id", "type"]]
        .dropna(subset=["race_id", "type"])
        .drop_duplicates(subset=["race_id"])
        .copy()
    )

    race_types["race_id"] = race_types["race_id"].astype(str)
    race_types["type"] = race_types["type"].astype(str).str.strip()

    print(f"Race types loaded: {len(race_types):,}")

    print("Loading Elo-scored dataset …")
    scored = read_csv_eu(SCORED_PATH)

    if "race_id" not in scored.columns:
        print("ERROR: scored dataset missing 'race_id'")
        print("Columns found:", list(scored.columns))
        return 2

    scored["race_id"] = scored["race_id"].astype(str)

    # Merge
    merged = scored.merge(
        race_types,
        on="race_id",
        how="left",
        validate="many_to_one",
    )

    missing = merged["type"].isna().sum()
    total = len(merged)

    print(f"Merged rows: {total:,}")
    print(f"Missing type after merge: {missing:,}")

    if missing > 0:
        print(
            "WARNING: Some rows did not get a type. "
            "These will remain NaN and should be inspected."
        )

    # Write output
    to_csv_eu(merged, OUTPUT_PATH)

    print(f"\nWrote: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
