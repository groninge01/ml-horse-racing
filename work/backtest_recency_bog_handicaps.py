#!/usr/bin/env python3
"""
backtest_recency_bog_handicaps.py

Backtest the recency+consistency signal in HANDICAPS for:
1) Flat turf handicaps (going != 'Standard')
2) Flat AW handicaps   (going == 'Standard')

Settlement uses BOG-style price:
    settle_price = max(OP_decimal, SP_decimal)
If OP missing -> SP.

Qualifier (pre-race):
- days_since_run in [0..5]
- prev_pos_1 <= 6
- prev_pos_2 <= 6
- surface consistency: current surface and prev surface both turf or both AW
- race_code == 'flat'
- handicap derived from race_name containing 'handicap'

Outputs EU CSVs to output/.
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


def frac_to_dec(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"", "-", "nan", "none"}:
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


def extract_op_frac(comment: object) -> str | None:
    if comment is None:
        return None
    s = str(comment).lower()
    m = OP_RE.search(s)
    if not m:
        return None
    return m.group(1).replace(" ", "")


def fetch(conn) -> pd.DataFrame:
    # comment may or may not exist in your table; we handle it after fetch.
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
      AND sp IS NOT NULL
    ORDER BY horse, date, race_id;
    """
    try:
        return pd.read_sql(sql, conn)
    except Exception:
        # If comment column doesn't exist, re-run without it.
        sql2 = """
        SELECT
            date        AS race_date,
            race_id,
            race_name,
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
          AND sp IS NOT NULL
        ORDER BY horse, date, race_id;
        """
        df = pd.read_sql(sql2, conn)
        df["comment"] = None
        return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")
    df["race_code"] = df["race_code"].astype(str)
    df["going"] = df["going"].astype(str)
    df["race_name"] = df["race_name"].astype(str)

    # SP odds
    df["sp_dec"] = df["sp"].apply(frac_to_dec)
    df = df.dropna(subset=["sp_dec"]).copy()

    # OP odds from comment (if any)
    df["op_frac"] = df["comment"].apply(extract_op_frac) if "comment" in df.columns else None
    df["op_dec"] = df["op_frac"].apply(frac_to_dec) if "op_frac" in df.columns else None

    # Handicap flag
    df["handicap"] = df["race_name"].str.contains("handicap", case=False, na=False)

    # Sort for lag features
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_key", sort=False)

    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)
    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)
    df["prev_going_1"] = g["going"].shift(1)

    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    return df


def backtest_block(df: pd.DataFrame, block_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (summary_df, by_oddsband_df) using BOG settle price = max(OP, SP).
    """
    df = df.copy()

    # Qualifier rule (pre-race)
    df["qualifier"] = (
        (df["days_since_run"].between(0, 5)) &
        (df["prev_pos_1"] <= 6) &
        (df["prev_pos_2"] <= 6)
    )

    # Surface consistency: current and previous must both be AW or both turf
    df["is_aw"] = (df["going"] == "Standard")
    df["prev_is_aw"] = (df["prev_going_1"] == "Standard")
    df["surface_ok"] = (df["is_aw"] == df["prev_is_aw"])

    # Restrict to bets (qualifiers only)
    bets = df[df["qualifier"] & df["surface_ok"]].copy()

    # Outcome
    bets["win"] = (bets["pos_int"] == 1).astype(int)

    # BOG settle price
    # If OP missing -> SP; else max(OP, SP)
    bets["settle_dec"] = bets[["sp_dec", "op_dec"]].max(axis=1, skipna=True)
    bets = bets.dropna(subset=["settle_dec"]).copy()

    # Profit per 1 unit stake
    bets["profit"] = bets["win"] * (bets["settle_dec"] - 1.0) - (1 - bets["win"])

    # Overall summary
    n = len(bets)
    wins = int(bets["win"].sum())
    profit = float(bets["profit"].sum())
    roi = profit / n if n else 0.0

    summary = pd.DataFrame([{
        "segment": block_name,
        "bets": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "avg_settle_odds": float(bets["settle_dec"].mean()) if n else 0.0,
        "profit": profit,
        "roi": roi,
        "op_available_share": float(bets["op_dec"].notna().mean()) if n else 0.0,
    }])

    # Odds band diagnostics on settle price
    bins = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1000.0]
    labels = ["<=2", "2-3", "3-5", "5-8", "8-12", "12+"]
    bets["odds_band"] = pd.cut(bets["settle_dec"], bins=bins, labels=labels, right=False)

    by_band = (
        bets.groupby("odds_band", observed=True)
        .agg(
            bets=("win", "size"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
            avg_settle_odds=("settle_dec", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    by_band["roi"] = by_band["profit"] / by_band["bets"]
    by_band.insert(0, "segment", block_name)

    return summary, by_band


def main():
    OUTDIR.mkdir(exist_ok=True)

    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        raw = fetch(conn)

    df = add_lags(raw)

    # Filter to FLAT handicaps only
    df = df[(df["race_code"].str.lower() == "flat") & (df["handicap"])].copy()

    # Split flat turf vs flat AW
    flat_turf = df[df["going"] != "Standard"].copy()
    flat_aw = df[df["going"] == "Standard"].copy()

    s1, b1 = backtest_block(flat_turf, "flat_turf_handicap_bog")
    s2, b2 = backtest_block(flat_aw, "flat_aw_handicap_bog")

    summary = pd.concat([s1, s2], ignore_index=True)
    by_band = pd.concat([b1, b2], ignore_index=True)

    summary_path = OUTDIR / "bog_backtest_flat_handicaps_summary.csv"
    bands_path = OUTDIR / "bog_backtest_flat_handicaps_by_oddsband.csv"

    summary.to_csv(summary_path, sep=";", decimal=",", index=False)
    by_band.to_csv(bands_path, sep=";", decimal=",", index=False)

    print("\n=== BOG BACKTEST (OP vs SP, take bigger) ===")
    print(summary.to_string(index=False))
    print("\nWrote:")
    print(f"- {summary_path}")
    print(f"- {bands_path}")


if __name__ == "__main__":
    main()
