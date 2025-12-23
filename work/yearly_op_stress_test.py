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

EDGE_THRESHOLD = 0.03   # model edge filter
TOP_K = 1               # top-1 selection

REQUIRED_COLS = {
    "race_date",
    "race_id",
    "horse_key",
    "y",
    "p_win",
}

# =========================
# ODDS PARSER (IMPORTANT)
# =========================
def parse_odds(val):
    """
    Converts Racing Post odds to decimal.
    Handles:
      9/2, 9/2F, 30/100, 11/4C, 6.0
    """
    if val is None:
        return np.nan

    s = str(val).strip()
    if s == "":
        return np.nan

    # remove trailing letters (F, C, J, etc)
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

    # model probability (EU decimal)
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
# FETCH PRICES (CLEAN JOIN)
# =========================
def fetch_prices(conn, race_ids):
    print("Fetching prices from Postgres…")

    sql = f"""
        SELECT
            race_id,
            horse_key,
            sp
        FROM {RUNNERS_TABLE}
        WHERE race_id = ANY(%s)
    """

    with conn.cursor() as cur:
        cur.execute(sql, (list(race_ids),))
        rows = cur.fetchall()

    prices = pd.DataFrame(rows, columns=["race_id", "horse_key", "sp"])
    prices["sp_dec"] = prices["sp"].apply(parse_odds)

    return prices


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 2:
        print("Usage: python yearly_op_stress_test.py <scored_csv>")
        sys.exit(1)

    scored_path = Path(sys.argv[1])
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)

    df = load_scored(scored_path)

    # =========================
    # MODEL EDGE VS IMPLIED OP
    # =========================
    # NOTE: we use SP implied prob as market proxy (OP not in DB)
    conn = psycopg.connect(DB_CONN_STR)
    prices = fetch_prices(conn, df["race_id"].unique())

    df = df.merge(prices, on=["race_id", "horse_key"], how="left")
    df = df.dropna(subset=["sp_dec"])

    df["p_market"] = 1.0 / df["sp_dec"]
    df["edge"] = df["p_model"] - df["p_market"]

    # =========================
    # TOP-K SELECTION PER RACE
    # =========================
    df["model_rank"] = df.groupby("race_id")["p_model"].rank(
        ascending=False, method="first"
    )

    df = df[
        (df["model_rank"] <= TOP_K) &
        (df["edge"] >= EDGE_THRESHOLD)
    ].copy()

    print(f"After filters: {len(df):,} bets | races: {df['race_id'].nunique():,}")

    # =========================
    # SETTLEMENT @ SP
    # =========================
    df["profit"] = np.where(
        df["y"] == 1,
        df["sp_dec"] - 1.0,
        -1.0
    )

    df["year"] = df["race_date"].dt.year

    yearly = (
        df.groupby("year")
        .agg(
            bets=("y", "count"),
            wins=("y", "sum"),
            win_rate=("y", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )

    yearly["roi"] = yearly["profit"] / yearly["bets"]

    out = Path("output")
    out.mkdir(exist_ok=True)
    yearly.to_csv(out / "yearly_op_stress_rank1.csv", index=False, sep=";")

    print("\n=== YEARLY STRESS TEST | TOP 1 | EDGE ≥ {:.2f} ===".format(EDGE_THRESHOLD))
    print(yearly)
    print("\nWrote: output/yearly_op_stress_rank1.csv")


if __name__ == "__main__":
    main()
