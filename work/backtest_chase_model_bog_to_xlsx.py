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
STAKE = 1.0

REQUIRED_COLS = {
    "race_date",
    "race_id",
    "horse_key",
    "y",
    "p_win",
}

# =========================
# ODDS PARSER
# =========================
def parse_odds(val):
    if val is None:
        return np.nan

    s = str(val).strip()
    if s == "":
        return np.nan

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
# FETCH CHASE META + PRICES
# =========================
def fetch_meta_and_prices(conn, race_ids):
    print("Fetching chase races + prices from Postgres…")

    sql = f"""
        SELECT
            race_id,
            off,
            course,
            horse_key,
            pos_int,
            sp,
            comment,
            type
        FROM {RUNNERS_TABLE}
        WHERE race_id = ANY(%s)
          AND type ILIKE '%%chase%%'
    """

    with conn.cursor() as cur:
        cur.execute(sql, (list(race_ids),))
        rows = cur.fetchall()

    meta = pd.DataFrame(
        rows,
        columns=[
            "race_id",
            "off",
            "course",
            "horse_key",
            "pos_int",
            "sp",
            "comment",
            "type",
        ],
    )

    meta["sp_dec"] = meta["sp"].apply(parse_odds)
    meta["op_dec"] = meta["comment"].apply(parse_odds)

    return meta


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 2:
        print("Usage: python backtest_chase_model_bog_to_xlsx.py <scored_csv>")
        sys.exit(1)

    scored_path = Path(sys.argv[1])
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)

    df = load_scored(scored_path)

    conn = psycopg.connect(DB_CONN_STR)
    meta = fetch_meta_and_prices(conn, df["race_id"].unique())

    df = df.merge(
        meta,
        on=["race_id", "horse_key"],
        how="inner"
    )

    df = df.dropna(subset=["sp_dec"])
    df["p_market"] = 1.0 / df["sp_dec"]
    df["edge"] = df["p_model"] - df["p_market"]

    df["model_rank"] = df.groupby("race_id")["p_model"].rank(
        ascending=False, method="first"
    )

    df = df[
        (df["model_rank"] <= TOP_K) &
        (df["edge"] >= EDGE_THRESHOLD)
    ].copy()

    print(f"After filters: {len(df):,} bets | races: {df['race_id'].nunique():,}")

    # =========================
    # BOG settlement
    # =========================
    df["settle_odds"] = np.where(
        df["op_dec"].notna(),
        np.maximum(df["op_dec"], df["sp_dec"]),
        df["sp_dec"],
    )

    df["stake"] = STAKE

    df["pnl"] = np.where(
        df["y"] == 1,
        (df["settle_odds"] - 1.0) * STAKE,
        -STAKE
    )

    df = df.sort_values(["race_date", "off", "course"])
    df["cum_pnl"] = df["pnl"].cumsum()

    out_df = df[
        [
            "race_date",
            "off",
            "course",
            "race_id",
            "horse_key",
            "pos_int",
            "p_model",
            "sp_dec",
            "op_dec",
            "settle_odds",
            "stake",
            "pnl",
            "cum_pnl",
        ]
    ].rename(columns={
        "horse_key": "horse",
        "pos_int": "finish_pos",
    })

    out = Path("output")
    out.mkdir(exist_ok=True)

    out_path = out / "chase_model_bog_backtest.xlsx"
    out_df.to_excel(out_path, index=False)

    print("\nWrote:")
    print(f" - {out_path.resolve()}")
    print(f"Total bets: {len(out_df):,}")
    print(f"Total PnL: {out_df['pnl'].sum():.2f}")


if __name__ == "__main__":
    main()
