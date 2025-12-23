#!/usr/bin/env python3
"""
Run SP calibration tests for the flat-recency signal across:

A) Flat turf (handicap / non-handicap)
B) All-weather flat (handicap / non-handicap)
C) Hurdle (handicap / non-handicap)
D) Chase (handicap / non-handicap)

NH Flat explicitly excluded.
"""

import pandas as pd
import psycopg
import re
from pathlib import Path

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
OUTDIR = Path("output")

FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


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


def fetch(conn) -> pd.DataFrame:
    sql = """
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
    return pd.read_sql(sql, conn)


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["pos_int"] = pd.to_numeric(df["pos_int"], errors="coerce")

    df["sp_dec"] = df["sp"].apply(frac_to_dec)
    df = df.dropna(subset=["sp_dec"])
    df["sp_implied"] = 1.0 / df["sp_dec"]

    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_key", sort=False)

    df["prev_date_1"] = g["race_date"].shift(1)
    df["prev_date_2"] = g["race_date"].shift(2)
    df["prev_pos_1"] = g["pos_int"].shift(1)
    df["prev_pos_2"] = g["pos_int"].shift(2)
    df["prev_going_1"] = g["going"].shift(1)

    df["days_since_run"] = (df["race_date"] - df["prev_date_1"]).dt.days

    # Core qualifier rule (surface filters applied later)
    df["qualifier"] = (
        (df["days_since_run"].between(0, 5)) &
        (df["prev_pos_1"] <= 6) &
        (df["prev_pos_2"] <= 6)
    )

    df["handicap"] = df["race_name"].str.contains("handicap", case=False, na=False)
    df["win"] = (df["pos_int"] == 1).astype(int)

    bins = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1000.0]
    labels = ["<=2", "2-3", "3-5", "5-8", "8-12", "12+"]
    df["odds_band"] = pd.cut(df["sp_dec"], bins=bins, labels=labels, right=False)

    return df


def run_block(df: pd.DataFrame, name: str):
    if df.empty:
        print(f"\n=== {name.upper()} === (no rows)")
        return

    out = (
        df.groupby(["odds_band", "qualifier"], observed=True)
        .agg(
            runners=("win", "size"),
            wins=("win", "sum"),
            actual_win_rate=("win", "mean"),
            implied_sp_prob=("sp_implied", "mean"),
        )
        .reset_index()
    )

    out["group"] = out["qualifier"].map({True: "qualifier", False: "non_qualifier"})
    out["delta"] = out["actual_win_rate"] - out["implied_sp_prob"]
    out = out.drop(columns=["qualifier"]).sort_values(["odds_band", "group"])

    OUTDIR.mkdir(exist_ok=True)
    out.to_csv(
        OUTDIR / f"{name}.csv",
        sep=";",
        decimal=",",
        index=False,
    )

    print(f"\n=== {name.upper()} ===")
    print(out.to_string(index=False))


def main():
    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        df = fetch(conn)

    df = prepare_base(df)

    # --------------------
    # A) Flat turf
    # --------------------
    flat_turf = df[
        (df["race_code"].str.lower() == "flat") &
        (df["going"] != "Standard")
    ]

    run_block(flat_turf[flat_turf["handicap"]], "calib_flat_turf_handicap")
    run_block(flat_turf[~flat_turf["handicap"]], "calib_flat_turf_non_handicap")

    # --------------------
    # B) All-weather flat
    # --------------------
    aw = df[
        (df["race_code"].str.lower() == "flat") &
        (df["going"] == "Standard")
    ]

    run_block(aw[aw["handicap"]], "calib_aw_handicap")
    run_block(aw[~aw["handicap"]], "calib_aw_non_handicap")

    # --------------------
    # C) Jumps (NO NH Flat)
    # --------------------
    hurdle = df[df["race_code"].str.lower() == "hurdle"]
    chase = df[df["race_code"].str.lower() == "chase"]

    run_block(hurdle[hurdle["handicap"]], "calib_hurdle_handicap")
    run_block(hurdle[~hurdle["handicap"]], "calib_hurdle_non_handicap")

    run_block(chase[chase["handicap"]], "calib_chase_handicap")
    run_block(chase[~chase["handicap"]], "calib_chase_non_handicap")


if __name__ == "__main__":
    main()
