#!/usr/bin/env python3
import sys
import re
import pandas as pd
import numpy as np
import psycopg
from pathlib import Path
import json

# =========================
# CONFIG
# =========================
DB_CONN_STR = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
SP_TABLE = "public.runners_ml"

REQUIRED_COLS = {
    "race_date",
    "race_id",
    "horse_key",
    "y",
    "p_win",
    "window_test_start",
}

# =========================
# SP PARSER (CRITICAL)
# =========================
def parse_sp_to_decimal(x):
    """
    Convert Racing Post SP formats to decimal odds.
    Handles:
      - 4/1
      - 30/100F
      - 11/8JF
      - decimal strings
    """
    if x is None:
        return np.nan

    s = str(x).strip().upper()

    # Remove suffixes like F, J, JF
    s = re.sub(r"(JF|F|J)$", "", s)

    # Fractional odds
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            return 1.0 + float(a) / float(b)
        except Exception:
            return np.nan

    # Decimal odds
    try:
        return float(s)
    except Exception:
        return np.nan


# =========================
# LOAD SCORED CSV
# =========================
def load_scored_csv(path: Path) -> pd.DataFrame:
    print("Loading scored test file…")

    df = pd.read_csv(path, sep=";")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Dates
    df["race_date"] = pd.to_datetime(
        df["race_date"], dayfirst=True, errors="coerce"
    )

    # Model probabilities (EU decimal)
    df["p_model"] = (
        df["p_win"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["y"] = df["y"].astype(int)
    df["race_id"] = df["race_id"].astype(int)
    df["horse_key"] = df["horse_key"].astype(str)

    df = df.dropna(subset=["race_date", "p_model"])

    print(
        f"Scored rows: {len(df):,} | unique races: {df['race_id'].nunique():,}"
    )
    return df


# =========================
# FETCH SP FROM POSTGRES
# =========================
import json

def fetch_sp(conn, keys: pd.DataFrame) -> pd.DataFrame:
    print("Fetching SP from Postgres…")

    sql = f"""
        SELECT
            race_id,
            horse_key,
            sp
        FROM {SP_TABLE}
        WHERE (race_id, horse_key) IN (
            SELECT
                (j->>'race_id')::bigint,
                j->>'horse_key'
            FROM jsonb_array_elements(%s::jsonb) AS j
        )
    """

    rows = []
    BATCH = 5000

    with conn.cursor() as cur:
        for i in range(0, len(keys), BATCH):
            chunk = keys.iloc[i : i + BATCH]
            payload = chunk.to_dict(orient="records")

            # 🔴 THIS WAS THE BUG
            cur.execute(sql, (json.dumps(payload),))

            rows.extend(cur.fetchall())

    sp = pd.DataFrame(rows, columns=["race_id", "horse_key", "sp"])

    sp["sp"] = sp["sp"].apply(parse_sp_to_decimal)
    sp = sp.dropna(subset=["sp"])

    return sp


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 2:
        print("Usage: python disagreement_analysis_sp.py <scored_csv>")
        sys.exit(1)

    scored_path = Path(sys.argv[1])
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)

    df = load_scored_csv(scored_path)

    conn = psycopg.connect(DB_CONN_STR)

    sp = fetch_sp(conn, df[["race_id", "horse_key"]].drop_duplicates())

    df = df.merge(sp, on=["race_id", "horse_key"], how="left")
    df = df.dropna(subset=["sp"])

    df["p_sp"] = 1.0 / df["sp"]
    df["edge"] = df["p_model"] - df["p_sp"]

    # Rank within race
    df["model_rank"] = df.groupby("race_id")["p_model"].rank(
        ascending=False, method="first"
    )
    df["sp_rank"] = df.groupby("race_id")["p_sp"].rank(
        ascending=False, method="first"
    )

    df["rank_gap"] = df["sp_rank"] - df["model_rank"]

    # =========================
    # OUTPUTS
    # =========================
    out = Path("output")
    out.mkdir(exist_ok=True)

    # Overall
    overall = pd.DataFrame(
        {
            "rows": [len(df)],
            "races": [df["race_id"].nunique()],
            "mean_edge": [df["edge"].mean()],
            "win_rate": [df["y"].mean()],
        }
    )
    overall.to_csv(out / "disagreement_overall.csv", index=False, sep=";")

    # By edge decile
    df["edge_decile"] = pd.qcut(df["edge"], 10, duplicates="drop")
    by_edge = (
        df.groupby("edge_decile", observed=True)
        .agg(
            runners=("y", "count"),
            winners=("y", "sum"),
            win_rate=("y", "mean"),
            mean_edge=("edge", "mean"),
            mean_sp=("sp", "mean"),
        )
        .reset_index()
    )
    by_edge.to_csv(out / "disagreement_by_edge.csv", index=False, sep=";")

    # Top-pick disagreement
    top = df[df["model_rank"] == 1]
    top_summary = (
        top.groupby(pd.cut(top["rank_gap"], [-50, -10, -5, 0, 5, 10, 50]))
        .agg(
            races=("race_id", "nunique"),
            wins=("y", "sum"),
            win_rate=("y", "mean"),
            mean_edge=("edge", "mean"),
        )
        .reset_index()
    )
    top_summary.to_csv(
        out / "disagreement_top1_rankgap.csv", index=False, sep=";"
    )

    print("\nWrote:")
    print(" - output/disagreement_overall.csv")
    print(" - output/disagreement_by_edge.csv")
    print(" - output/disagreement_top1_rankgap.csv")


if __name__ == "__main__":
    main()
