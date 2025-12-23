#!/usr/bin/env python3
import sys
import re
import numpy as np
import pandas as pd
import psycopg
from pathlib import Path

# =========================
# CONFIG
# =========================
DB_CONN_STR = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
RUNNERS_TABLE = "public.runners_ml"
CSV_SEP = ";"

EDGE_THRESHOLD = 0.03
TOP_K = 1

REQUIRED_COLS = {
    "race_date",
    "race_id",
    "horse_key",
    "y",
    "p_win",
}

# =========================
# ODDS PARSER (SAFE)
# =========================
def parse_odds(val):
    if val is None:
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan

    # strip trailing letters (F, C, J, etc)
    s = re.sub(r"[A-Za-z]+$", "", s)

    if "/" in s:
        try:
            a, b = s.split("/")
            return float(a) / float(b) + 1.0
        except Exception:
            return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


# =========================
# LOAD SCORED CSV
# =========================
def load_scored(path: Path) -> pd.DataFrame:
    print("Loading scored test file…")
    df = pd.read_csv(path, sep=CSV_SEP)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["race_date"] = pd.to_datetime(df["race_date"], dayfirst=True, errors="coerce")

    df["p_model"] = (
        df["p_win"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["race_id"] = df["race_id"].astype(int)
    df["horse_key"] = df["horse_key"].astype(str)
    df["y"] = df["y"].astype(int)

    df = df.dropna(subset=["race_date", "p_model"])

    print(f"Scored rows: {len(df):,} | races: {df['race_id'].nunique():,}")
    return df


# =========================
# FETCH META + SP (RACE-SAFE)
# =========================
def fetch_meta_and_sp(conn, race_ids):
    print("Fetching race meta + SP from Postgres…")

    sql = f"""
        SELECT
            race_id,
            horse_key,
            sp,
            CASE
                WHEN type ILIKE '%%chase%%' THEN 'chase'
                WHEN type ILIKE '%%hurdle%%' THEN 'hurdle'
                WHEN type ILIKE '%%aw%%' THEN 'flat_aw'
                ELSE 'flat_turf'
            END AS surface,
            CASE
                WHEN type ILIKE '%%hcp%%'
                  OR race_name ILIKE '%%handicap%%'
                THEN TRUE
                ELSE FALSE
            END AS handicap
        FROM {RUNNERS_TABLE}
        WHERE race_id = ANY(%s)
    """

    meta = pd.read_sql(
        sql,
        conn,
        params=(list(race_ids),)
    )

    meta["sp_dec"] = meta["sp"].apply(parse_odds)
    return meta

# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 2:
        print("Usage: python segment_op_stress_total.py <scored_csv>")
        sys.exit(1)

    scored_path = Path(sys.argv[1])
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)

    df = load_scored(scored_path)

    # =========================
    # MODEL RANK (RACE-LEVEL)
    # =========================
    df["model_rank"] = df.groupby("race_id")["p_model"].rank(
        ascending=False, method="first"
    )

    # ENFORCE: exactly one runner per race
    df = df[df["model_rank"] <= TOP_K].copy()

    # =========================
    # JOIN MARKET DATA
    # =========================
    conn = psycopg.connect(DB_CONN_STR)
    meta = fetch_meta_and_sp(conn, df["race_id"].unique())

    df = df.merge(
        meta,
        on=["race_id", "horse_key"],
        how="left"
    )

    df = df.dropna(subset=["sp_dec"])

    # =========================
    # EDGE FILTER (IDENTICAL TO YEARLY)
    # =========================
    df["p_market"] = 1.0 / df["sp_dec"]
    df["edge"] = df["p_model"] - df["p_market"]

    df = df[df["edge"] >= EDGE_THRESHOLD].copy()

    print(f"After filters: {len(df):,} bets | races: {df['race_id'].nunique():,}")

    # =========================
    # SETTLEMENT @ SP
    # =========================
    df["profit"] = np.where(
        df["y"] == 1,
        df["sp_dec"] - 1.0,
        -1.0
    )

    # =========================
    # SEGMENT SUMMARY (TOTAL)
    # =========================
    summary = (
        df.groupby(["surface", "handicap"])
        .agg(
            bets=("y", "count"),
            wins=("y", "sum"),
            win_rate=("y", "mean"),
            avg_sp=("sp_dec", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )

    summary["roi"] = summary["profit"] / summary["bets"]

    out = Path("output")
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "segment_op_stress_total.csv", index=False, sep=";")

    print("\n=== SEGMENT TOTAL SUMMARY (RACE-LEVEL, SAME AS YEARLY) ===")
    print(summary)
    print("\nWrote: output/segment_op_stress_total.csv")


if __name__ == "__main__":
    main()
