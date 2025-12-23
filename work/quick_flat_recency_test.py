#!/usr/bin/env python3
"""
quick_flat_recency_test.py

Pattern test (PRE-RACE ONLY):

1) Flat turf races (AW excluded via going == 'Standard')
2) Horse ran within 5 days
3) Last run was also turf
4) Finished top 6 on BOTH last two starts

Data source: public.runners_raw
"""

import pandas as pd
import psycopg
from pathlib import Path

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
OUTDIR = Path("output")


def fetch(conn) -> pd.DataFrame:
    sql = """
    SELECT
        date        AS race_date,
        race_id,
        course,
        type        AS race_code,
        going,
        horse       AS horse_key,
        pos         AS pos_int
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

    print(f"Runner rows: {len(df):,}")
    print(f"Races: {df['race_id'].nunique():,}")

    # Basic cleaning
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")
    df["going"] = df["going"].astype(str)
    df["race_code"] = df["race_code"].astype(str)

    # Sort for lag logic
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)

    g = df.groupby("horse_key", sort=False)

    # Lag features
    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)

    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)

    df["prev_going_1"] = g["going"].shift(1)

    # Days since last run
    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    # Strategy filters
    mask = (
        (df["race_code"].str.lower() == "flat") &      # flat only
        (df["going"] != "Standard") &                  # turf only (current)
        (df["days_since_run"].between(0, 5)) &         # ran within 5 days
        (df["prev_going_1"] != "Standard") &           # last run turf
        (df["prev_pos_1"] <= 6) &                      # top 6 last run
        (df["prev_pos_2"] <= 6)                        # top 6 run before
    )

    qual = df[mask].copy()
    qual["win"] = (qual["pos_int"] == 1).astype(int)
    qual["place"] = (qual["pos_int"] <= 3).astype(int)

    # Baseline: all runners in the same races
    base = df[df["race_id"].isin(qual["race_id"])].copy()
    base["win"] = (base["pos_int"] == 1).astype(int)

    # Metrics
    n = len(qual)
    wins = int(qual["win"].sum())
    places = int(qual["place"].sum())

    win_rate = wins / n if n else 0.0
    place_rate = places / n if n else 0.0
    base_win_rate = base["win"].mean()
    iv = win_rate / base_win_rate if base_win_rate > 0 else float("nan")

    summary = pd.DataFrame([{
        "qualifiers": n,
        "wins": wins,
        "win_rate": win_rate,
        "places": places,
        "place_rate": place_rate,
        "baseline_win_rate": base_win_rate,
        "impact_value": iv,
    }])

    OUTDIR.mkdir(exist_ok=True)
    summary.to_csv(
        OUTDIR / "flat_recency_strategy_summary.csv",
        sep=";",
        decimal=",",
        index=False,
    )

    print("\n=== FLAT TURF RECENCY STRATEGY ===")
    print(summary.to_string(index=False))
    print(f"\nWrote: {OUTDIR / 'flat_recency_strategy_summary.csv'}")


if __name__ == "__main__":
    main()
