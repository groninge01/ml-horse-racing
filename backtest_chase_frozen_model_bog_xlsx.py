#!/usr/bin/env python3
import re
import joblib
import numpy as np
import pandas as pd
import psycopg
from pathlib import Path
from catboost import CatBoostClassifier

from build_form_features_and_train import clean_base, add_lag_features, build_feature_matrix

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
RUNNERS = "public.runners_ml"

MODEL_PATH = Path("frozen_model/chase_model_apr2025.cbm")
FEATURES_PATH = Path("frozen_model/features_apr2025.joblib")

OUT_XLSX = Path("output/chase_model_bog_backtest.xlsx")
STAKE = 1.0


# ---------- Odds parsing ----------

def parse_odds(val):
    if val is None:
        return np.nan
    s = str(val).strip()
    s = re.sub(r"[A-Za-z]+$", "", s)
    if "/" in s:
        try:
            a, b = s.split("/")
            return float(a) / float(b) + 1
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def extract_op(comment):
    if not isinstance(comment, str):
        return np.nan
    m = re.search(r"(\d+/\d+)", comment)
    return parse_odds(m.group(1)) if m else np.nan


# ---------- Main ----------

def main():
    print("Loading frozen model…")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    print("Fetching raw chase runners…")

    sql = f"""
        SELECT
            race_date,
            race_id,
            off,
            course,
            race_name,
            type,
            going,
            class,
            horse_key,
            age,
            sex,
            or_rating,
            draw,
            jockey,
            trainer,
            pos_int,
            sp,
            comment
        FROM {RUNNERS}
        WHERE race_date > '2025-04-30'
          AND type = 'Chase'
          AND pos_int IS NOT NULL
        ORDER BY race_date, race_id
    """

    with psycopg.connect(DSN) as conn:
        raw = pd.read_sql(sql, conn)

    print(f"Rows fetched: {len(raw):,}")

    # Rebuild feature pipeline exactly as training
    df = clean_base(raw)
    df = add_lag_features(df)

    # Filter to only post-cutoff rows again after lags
    df = df[df["race_date"] > pd.Timestamp("2025-04-30")].copy()

    # Parse prices
    df["sp_dec"] = df["sp"].apply(parse_odds)
    df["op_dec"] = df["comment"].apply(extract_op)
    df["settle_odds"] = df[["sp_dec", "op_dec"]].max(axis=1)

    # Build model matrix
    X, _, cat_idx, _ = build_feature_matrix(df)

    # Predict
    df["model_prob"] = model.predict_proba(X)[:, 1]

    # Select top pick per race
    df["model_rank"] = df.groupby("race_id")["model_prob"].rank(
        ascending=False, method="first"
    )
    bets = df[df["model_rank"] == 1].copy()

    # Settlement
    bets["stake"] = STAKE
    bets["pnl"] = np.where(
        bets["pos_int"] == 1,
        bets["settle_odds"] - 1,
        -1,
    )
    bets["cum_pnl"] = bets["pnl"].cumsum()

    out = bets[
        [
            "race_date",
            "off",
            "course",
            "race_id",
            "horse_key",
            "model_prob",
            "op_dec",
            "sp_dec",
            "settle_odds",
            "pos_int",
            "stake",
            "pnl",
            "cum_pnl",
        ]
    ].copy()

    # ---------- ONLY CHANGE: sort output ----------

    out["off_time"] = pd.to_datetime(out["off"], errors="coerce").dt.time
    out = out.sort_values(["race_date", "off_time", "course", "race_id"])
    out = out.drop(columns=["off_time"])

    # --------------------------------------------

    OUT_XLSX.parent.mkdir(exist_ok=True)
    out.to_excel(OUT_XLSX, index=False)

    print(f"\nWrote: {OUT_XLSX}")
    print(f"Bets: {len(out)} | Total PnL: {out['pnl'].sum():.2f}")


if __name__ == "__main__":
    main()
