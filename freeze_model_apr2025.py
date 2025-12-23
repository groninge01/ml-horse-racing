#!/usr/bin/env python3
import joblib
from pathlib import Path
import pandas as pd
import psycopg
from catboost import CatBoostClassifier

from build_form_features_and_train import (
    choose_source,
    fetch_runner_rows,
    clean_base,
    add_lag_features,
    build_feature_matrix,
    CATBOOST_PARAMS,
)

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
CUTOFF_DATE = pd.Timestamp("2025-04-30")
OUTDIR = Path("frozen_model")
OUTDIR.mkdir(exist_ok=True)

MODEL_PATH = OUTDIR / "chase_model_apr2025.cbm"
FEATURES_PATH = OUTDIR / "features_apr2025.joblib"

def main():
    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        schema, name = choose_source(conn)
        raw = fetch_runner_rows(conn, schema, name)

    raw["race_date"] = pd.to_datetime(raw["race_date"], errors="coerce")
    raw = raw[raw["race_date"] <= CUTOFF_DATE]

    print(f"Training rows: {len(raw):,}")

    df = clean_base(raw)
    df = add_lag_features(df)

    X, y, cat_idx, feature_cols = build_feature_matrix(df)

    print("Training frozen model…")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X, y, cat_features=cat_idx)

    model.save_model(MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    print("\nFrozen model written:")
    print(f" - {MODEL_PATH}")
    print(f" - {FEATURES_PATH}")

if __name__ == "__main__":
    main()
