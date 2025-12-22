#!/usr/bin/env python3
"""
horse_ml.py

Baseline horse racing ML model:
- Predicts WIN (pos == 1)
- Time-based train/test split
- CatBoost with categorical support
- Adds market-derived features:
    - sp_implied
    - sp_rank
    - is_favorite
- Runs a simple SP-based value betting simulation

Usage:
  python horse_ml.py races.csv
"""

import sys
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
from catboost import CatBoostClassifier, Pool


# =============================
# Text / parsing helpers
# =============================

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def parse_weight_lbs(w: str) -> Optional[float]:
    if w is None or (isinstance(w, float) and np.isnan(w)):
        return None
    w = normalize_text(w)
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", w)
    if not m:
        return None
    return int(m.group(1)) * 14 + int(m.group(2))

def parse_fractional_sp_to_decimal(sp: str) -> Optional[float]:
    if sp is None or (isinstance(sp, float) and np.isnan(sp)):
        return None
    sp = normalize_text(sp).upper()
    sp = re.sub(r"[A-Z]+$", "", sp)  # strip F, J, C, etc.

    if sp in ("EVS", "EVENS"):
        return 2.0

    m = re.match(r"^(\d+)\s*/\s*(\d+)$", sp)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    return 1.0 + (num / den)

def dist_to_furlongs(dist: str) -> Optional[float]:
    if dist is None or (isinstance(dist, float) and np.isnan(dist)):
        return None
    s = normalize_text(dist).replace("Â½", "½")
    miles, furlongs = 0.0, 0.0

    mm = re.search(r"(\d+)\s*m", s)
    if mm:
        miles = float(mm.group(1))

    mf = re.search(r"(\d+)(½)?\s*f", s)
    if mf:
        furlongs = float(mf.group(1)) + (0.5 if mf.group(2) else 0.0)

    return miles * 8.0 + furlongs

def pos_is_win(pos) -> Optional[int]:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return None
    s = normalize_text(pos)
    try:
        return 1 if int(float(s)) == 1 else 0
    except Exception:
        return 0


# =============================
# Betting simulation
# =============================

def simulate_value_bets(df, p_col, odds_col, margin=0.05, stake=1.0):
    d = df.dropna(subset=[p_col, odds_col, "y"])
    if d.empty:
        return 0, 0.0, 0.0

    implied = 1.0 / d[odds_col]
    bets = d[d[p_col] > implied * (1 + margin)]

    if bets.empty:
        return 0, 0.0, 0.0

    wins = bets["y"] == 1
    profit = np.where(
        wins,
        (bets[odds_col] - 1.0) * stake,
        -stake
    ).sum()

    roi = profit / (len(bets) * stake)
    strike = wins.mean()

    return len(bets), strike, roi


# =============================
# Main
# =============================

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def main(path):
    df = read_csv_safe(path)
    df.columns = [normalize_text(c) for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Core parsing
    df["wgt_lbs"] = df["wgt"].apply(parse_weight_lbs)
    df["dist_f"] = df["dist"].apply(dist_to_furlongs)
    df["sp_dec"] = df["sp"].apply(parse_fractional_sp_to_decimal)
    df["y"] = df["pos"].apply(pos_is_win)

    df = df.dropna(subset=["y", "sp_dec", "dist_f", "wgt_lbs"])
    df["y"] = df["y"].astype(int)

    # Market-derived features
    df["sp_implied"] = 1.0 / df["sp_dec"]
    df["sp_rank"] = (
        df.groupby("race_id")["sp_dec"]
          .rank(method="min", ascending=True)
    )
    df["is_favorite"] = (df["sp_rank"] == 1).astype(int)

    # Numeric coercions
    for col in ["ran", "draw", "age", "or", "rpr", "ts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature set
    features = [
        "course", "race_id", "type", "class",
        "dist_f", "going", "ran", "draw", "age", "sex",
        "wgt_lbs",
        "jockey", "trainer",
        "or", "rpr", "ts",
        "sp_dec", "sp_implied", "sp_rank", "is_favorite"
    ]
    features = [f for f in features if f in df.columns]

    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[features], train["y"]
    X_test, y_test = test[features], test["y"]

    cat_cols = [c for c in features if X_train[c].dtype == "object"]
    cat_idx = [features.index(c) for c in cat_cols]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1200,
        depth=8,
        learning_rate=0.08,
        random_seed=42,
        verbose=200,
        thread_count=-1
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    p_test = model.predict_proba(test_pool)[:, 1]
    test["p_win"] = p_test

    print("\n=== MODEL METRICS (OOS) ===")
    print(f"AUC:     {roc_auc_score(y_test, p_test):.4f}")
    print(f"LogLoss: {log_loss(y_test, p_test):.4f}")

    print("\n=== CALIBRATION (quantiles) ===")
    frac, mean = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    for i, (m, f) in enumerate(zip(mean, frac), 1):
        print(f"Bin {i:02d}: pred={m:.3f}  actual={f:.3f}")

    print("\n=== VALUE BET SIMULATION (SP) ===")
    for m in (0.02, 0.05, 0.08, 0.10):
        n, sr, roi = simulate_value_bets(test, "p_win", "sp_dec", margin=m)
        print(f"Edge {m:0.2f}: bets={n:5d} strike={sr:0.3f} ROI={roi:0.3f}")

    print("\n=== TOP FEATURE IMPORTANCE ===")
    fi = model.get_feature_importance(train_pool)
    fi_df = (
        pd.DataFrame({"feature": features, "importance": fi})
          .sort_values("importance", ascending=False)
          .head(20)
    )
    print(fi_df.to_string(index=False))

    out = test[
        ["date", "course", "race_id", "horse", "sp", "sp_dec", "is_favorite", "p_win", "y"]
    ]
    out.to_csv("horse_ml_scored_test.csv", index=False)
    print("\nWrote: horse_ml_scored_test.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python horse_ml.py races.csv")
    main(sys.argv[1])
