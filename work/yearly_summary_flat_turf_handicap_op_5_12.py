#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import psycopg

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
OUTDIR = Path("output")

FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
OP_RE = re.compile(r"\bop\s+(\d+\s*/\s*\d+)\b", re.IGNORECASE)


def frac_to_dec(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"", "-", "nan"}:
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


def extract_op_frac(comment):
    if comment is None:
        return None
    m = OP_RE.search(str(comment).lower())
    if not m:
        return None
    return m.group(1).replace(" ", "")


def fetch(conn) -> pd.DataFrame:
    sql = """
    SELECT
        date        AS race_date,
        race_id,
        race_name,
        type        AS race_type,
        going,
        horse       AS horse_key,
        pos         AS pos_int,
        comment
    FROM public.runners_raw
    WHERE date IS NOT NULL
      AND race_id IS NOT NULL
      AND horse IS NOT NULL
      AND pos IS NOT NULL
      AND comment IS NOT NULL
    ORDER BY horse, date, race_id;
    """
    return pd.read_sql(sql, conn)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["year"] = df["race_date"].dt.year
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")

    df["op_frac"] = df["comment"].apply(extract_op_frac)
    df["op_dec"] = df["op_frac"].apply(frac_to_dec)

    df = df.dropna(subset=["race_date", "pos_int", "op_dec"]).copy()

    df["handicap"] = df["race_name"].str.contains("handicap", case=False, na=False)

    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_key", sort=False)

    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)
    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)
    df["prev_going_1"] = g["going"].shift(1)
    df["prev_type_1"] = g["race_type"].shift(1)
    df["prev_type_2"] = g["race_type"].shift(2)

    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    return df


def main():
    OUTDIR.mkdir(exist_ok=True)

    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        df = fetch(conn)

    df = prepare(df)

    # --- Universe: flat turf handicaps ---
    d = df.copy()
    d = d[d["handicap"]]
    d = d[d["race_type"].str.lower() == "flat"]
    d = d[d["going"] != "Standard"]
    d = d[d["prev_going_1"] != "Standard"]
    d = d[d["prev_type_1"].str.lower() == "flat"]
    d = d[d["prev_type_2"].str.lower() == "flat"]

    # --- Qualifier ---
    d = d[
        d["days_since_run"].between(0, 5) &
        (d["prev_pos_1"] <= 6) &
        (d["prev_pos_2"] <= 6)
    ].copy()

    # --- OP odds filter: 5–12 ---
    d = d[(d["op_dec"] >= 5.0) & (d["op_dec"] < 12.0)].copy()

    # --- Outcome & PnL ---
    d["win"] = (d["pos_int"] == 1).astype(int)
    d["profit"] = d["win"] * (d["op_dec"] - 1.0) - (1 - d["win"])

    yearly = (
        d.groupby("year", observed=True)
        .agg(
            bets=("win", "size"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
            avg_op_odds=("op_dec", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )

    yearly["roi"] = yearly["profit"] / yearly["bets"]

    out = OUTDIR / "yearly_flat_turf_handicap_op_5_12.csv"
    yearly.to_csv(out, sep=";", decimal=",", index=False)

    print("\n=== YEARLY SUMMARY: FLAT TURF HANDICAP (OP 5–12) ===")
    print(yearly.to_string(index=False))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
