#!/usr/bin/env python3
"""
build_form_features_and_train.py

Hard reset starter:
- Load runner rows from Postgres (auto-detect best source)
- Build strictly pre-race lag features (last N races per horse)
- Train CatBoost winner classifier
- Rolling forward evaluation (train 18 months, test 6 months)
- Output EU CSVs to output/

No odds. No Elo. No post-race features except the label (pos==1).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg
from catboost import CatBoostClassifier

DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
OUTDIR = Path("output")

# Rolling protocol
TRAIN_MONTHS = 18
TEST_MONTHS = 6

# Form window per horse (last N races)
N_LAG = 6

# CatBoost settings (CPU-friendly, adjust if needed)
CATBOOST_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5.0,
    random_seed=42,
    od_type="Iter",
    od_wait=100,
    verbose=200,
    allow_writing_files=False,
)


# ----------------------------
# EU CSV helper
# ----------------------------
def to_csv_eu(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", decimal=",", date_format="%d-%m-%Y")


# ----------------------------
# Postgres discovery utilities
# ----------------------------
def pg_has_relation(conn, schema: str, name: str) -> bool:
    sql = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = %s AND table_name = %s
    UNION ALL
    SELECT 1
    FROM information_schema.views
    WHERE table_schema = %s AND table_name = %s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, name, schema, name))
        return cur.fetchone() is not None


def pg_columns(conn, schema: str, name: str) -> List[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, name))
        return [r[0] for r in cur.fetchall()]


def choose_source(conn) -> Tuple[str, str]:
    """
    Prefer a rich runner-level view if it exists.
    In your project, public.runners_ml has the right pre-race runner rows.
    """
    candidates = [
        ("public", "runners_ml"),
    ]
    for schema, name in candidates:
        if pg_has_relation(conn, schema, name):
            return schema, name
    raise RuntimeError("Could not find a suitable source view/table (tried: runners_ml).")


def fetch_runner_rows(conn, schema: str, name: str) -> pd.DataFrame:
    cols = set(pg_columns(conn, schema, name))

    # Required minimal set for this restart:
    required_any = {"race_date", "race_id", "horse_key", "pos_int"}
    if not required_any.issubset(cols):
        raise RuntimeError(
            f"Source {schema}.{name} missing required columns {sorted(required_any)}. "
            f"Available columns: {sorted(cols)[:60]}..."
        )

    # Optional columns we'd like if present
    want = [
        "race_date", "race_id", "course", "off", "race_name",
        "type", "race_code", "handicap",
        "ran_int", "going", "dist_f", "class",
        "horse_key", "horse", "age", "sex",
        "wgt_lbs", "or_rating", "draw",
        "jockey", "trainer",
        "pos_int",
    ]

    select_cols = [c for c in want if c in cols]
    # Ensure required are included
    for r in ["race_date", "race_id", "horse_key", "pos_int"]:
        if r not in select_cols:
            select_cols.append(r)

    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM {schema}.{name}
    WHERE race_date IS NOT NULL
      AND race_id IS NOT NULL
      AND horse_key IS NOT NULL
      AND pos_int IS NOT NULL
    ORDER BY race_date, race_id;
    """
    df = pd.read_sql(sql, conn)
    return df


# ----------------------------
# Feature engineering (strictly pre-race)
# ----------------------------
def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date", "race_id", "horse_key", "pos_int"])

    # Label: winner
    df["y"] = (pd.to_numeric(df["pos_int"], errors="coerce") == 1).astype(int)

    # Field size
    if "ran_int" in df.columns:
        df["field_size"] = pd.to_numeric(df["ran_int"], errors="coerce")
    else:
        df["field_size"] = df.groupby("race_id")["horse_key"].transform("count").astype(float)

    # Numeric cleaning
    for col in ["age", "wgt_lbs", "or_rating", "draw", "field_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic cats - keep as strings
    for c in ["course", "type", "race_code", "going", "sex", "jockey", "trainer", "race_name", "class", "handicap"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": None, "None": None})

    return df


def add_lag_features(df: pd.DataFrame, n_lag: int = N_LAG) -> pd.DataFrame:
    """
    Create strictly pre-race lag features by horse_key using past races only.
    IMPORTANT: we sort by (race_date, race_id) and shift BEFORE rolling.
    """
    df = df.sort_values(["horse_key", "race_date", "race_id"]).reset_index(drop=True).copy()

    g = df.groupby("horse_key", sort=False)

    # Days since last run
    prev_date = g["race_date"].shift(1)
    df["days_since_run"] = (df["race_date"] - prev_date).dt.days.astype(float)

    # Previous outcomes / stats
    prev_y = g["y"].shift(1)
    df["won_last"] = prev_y.astype(float)

    # Last finish position (pos_int) as numeric, shifted
    prev_pos = pd.to_numeric(g["pos_int"].shift(1), errors="coerce")
    df["last_pos"] = prev_pos

    # Rolling win rate over last N races
    df["win_rate_lagN"] = g["y"].shift(1).rolling(n_lag, min_periods=1).mean().reset_index(level=0, drop=True)

    # Rolling average last_pos (lower is better)
    df["avg_pos_lagN"] = prev_pos.groupby(df["horse_key"]).rolling(n_lag, min_periods=1).mean().reset_index(level=0, drop=True)

    # Rolling runs count
    df["runs_lagN"] = g["y"].shift(1).rolling(n_lag, min_periods=1).count().reset_index(level=0, drop=True)

    # OR trend and rolling OR
    if "or_rating" in df.columns:
        prev_or = g["or_rating"].shift(1)
        prev2_or = g["or_rating"].shift(2)
        df["last_or"] = pd.to_numeric(prev_or, errors="coerce")
        df["or_change_last"] = pd.to_numeric(prev_or, errors="coerce") - pd.to_numeric(prev2_or, errors="coerce")
        df["or_mean_lagN"] = g["or_rating"].shift(1).rolling(n_lag, min_periods=1).mean().reset_index(level=0, drop=True)

    # Weight trend
    if "wgt_lbs" in df.columns:
        prev_w = g["wgt_lbs"].shift(1)
        prev2_w = g["wgt_lbs"].shift(2)
        df["last_wgt"] = pd.to_numeric(prev_w, errors="coerce")
        df["wgt_change_last"] = pd.to_numeric(prev_w, errors="coerce") - pd.to_numeric(prev2_w, errors="coerce")
        df["wgt_mean_lagN"] = g["wgt_lbs"].shift(1).rolling(n_lag, min_periods=1).mean().reset_index(level=0, drop=True)

    # Last seen categorical contexts (shifted)
    for cat in ["type", "going", "course", "trainer", "jockey", "race_code"]:
        if cat in df.columns:
            df[f"last_{cat}"] = g[cat].shift(1)

    # Horse career runs to date (pre-race)
    df["career_runs_pre"] = g.cumcount().astype(int)

    # Drop rows with no history? For modelling, we keep them, but they’ll have NaNs.
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[int], List[str]]:
    """
    Choose a clean feature set. No odds, no Elo.
    """
    base_cols = ["race_date", "race_id", "horse_key", "y"]

    feature_cols = []

    # Current-race numeric
    for c in ["age", "wgt_lbs", "or_rating", "draw", "field_size"]:
        if c in df.columns:
            feature_cols.append(c)

    # Current-race categorical
    for c in ["course", "type", "race_code", "going", "sex", "trainer", "jockey", "handicap", "class"]:
        if c in df.columns:
            feature_cols.append(c)

    # Lagged numeric
    for c in [
        "days_since_run", "won_last", "last_pos",
        "win_rate_lagN", "avg_pos_lagN", "runs_lagN",
        "career_runs_pre",
        "last_or", "or_change_last", "or_mean_lagN",
        "last_wgt", "wgt_change_last", "wgt_mean_lagN",
    ]:
        if c in df.columns:
            feature_cols.append(c)

    # Lagged categorical
    for c in ["last_type", "last_going", "last_course", "last_trainer", "last_jockey", "last_race_code"]:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].copy()
    y = df["y"].astype(int)

    # CatBoost categorical indices
    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    # Convert cats to string and fill NA (CatBoost cannot handle NaN in categorical)
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("NA").astype(str)

    # Numeric columns: keep NaN (CatBoost can handle)
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return X, y, cat_idx, feature_cols


# ----------------------------
# Metrics (race-level)
# ----------------------------
def runner_logloss(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def runner_auc(p: np.ndarray, y: np.ndarray) -> float:
    # rank-based AUC without sklearn
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    y_sorted = y[order]
    n1 = int(y_sorted.sum())
    n0 = len(y_sorted) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    ranks = np.arange(1, len(y_sorted) + 1)
    sum_ranks_pos = float(np.sum(ranks[y_sorted == 1]))
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n0 * n1)
    return float(auc)


def top1_accuracy(df_test: pd.DataFrame, p: np.ndarray) -> float:
    tmp = df_test[["race_id", "y"]].copy()
    tmp["p"] = p
    # pick max p per race; check winner label
    idx = tmp.groupby("race_id")["p"].idxmax()
    return float(tmp.loc[idx, "y"].mean())


def race_logloss(df_test: pd.DataFrame, p: np.ndarray) -> float:
    """
    Convert per-runner scores to per-race probabilities by normalising within race:
      p_race_i = p_i / sum_j p_j
    then logloss = mean(-log(prob_winner))
    """
    eps = 1e-15
    tmp = df_test[["race_id", "y"]].copy()
    tmp["p"] = np.clip(p, eps, 1.0)
    denom = tmp.groupby("race_id")["p"].transform("sum")
    tmp["p_norm"] = tmp["p"] / np.clip(denom, eps, None)
    pw = tmp.groupby("race_id").apply(lambda g: float((g["p_norm"] * g["y"]).sum()))
    return float((-np.log(np.clip(pw.values, eps, 1.0))).mean())


# ----------------------------
# Rolling train/test
# ----------------------------
def rolling_windows(df: pd.DataFrame) -> List[Dict]:
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    start = df["race_date"].min()
    end = df["race_date"].max()

    windows = []
    cur = start
    while True:
        train_end = cur + pd.DateOffset(months=TRAIN_MONTHS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        train = df[(df["race_date"] >= cur) & (df["race_date"] < train_end)]
        test = df[(df["race_date"] >= train_end) & (df["race_date"] < test_end)]
        if test.empty:
            break

        # sanity: enough races
        if train["race_id"].nunique() < 5000 or test["race_id"].nunique() < 1000:
            cur = train_end
            if cur >= end:
                break
            continue

        windows.append(dict(
            train_start=cur.date(),
            train_end=train_end.date(),
            test_start=train_end.date(),
            test_end=test_end.date(),
            train_rows=len(train),
            test_rows=len(test),
            train_races=int(train["race_id"].nunique()),
            test_races=int(test["race_id"].nunique()),
            train_df=train,
            test_df=test,
        ))
        cur = train_end
        if cur >= end:
            break

    return windows


def main() -> int:
    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        schema, name = choose_source(conn)
        print(f"Using source: {schema}.{name}")

        print("Fetching runner rows…")
        raw = fetch_runner_rows(conn, schema, name)

    print(f"Rows fetched: {len(raw):,}")
    print(f"Races: {raw['race_id'].nunique():,}")

    df = clean_base(raw)
    df = add_lag_features(df, n_lag=N_LAG)

    # Sort globally for windowing
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    # Build windows
    wins = rolling_windows(df)
    if not wins:
        print("No rolling windows formed (insufficient time span or too few races).")
        return 0

    all_window_metrics = []
    all_scored_rows = []

    for w in wins:
        train = w.pop("train_df")
        test = w.pop("test_df")

        # Build feature matrices
        X_train, y_train, cat_idx, feature_cols = build_feature_matrix(train)
        X_test, y_test, _, _ = build_feature_matrix(test)

        print("\n=== WINDOW ===")
        print(f"Train {w['train_start']} → {w['train_end']}  rows={w['train_rows']:,} races={w['train_races']:,}")
        print(f"Test  {w['test_start']} → {w['test_end']}  rows={w['test_rows']:,} races={w['test_races']:,}")

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            cat_features=cat_idx,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )

        p_test = model.predict_proba(X_test)[:, 1]
        auc = runner_auc(p_test, y_test.to_numpy(dtype=int))
        ll = runner_logloss(p_test, y_test.to_numpy(dtype=int))
        t1 = top1_accuracy(test, p_test)
        rll = race_logloss(test, p_test)

        metrics = dict(
            **w,
            runner_auc=auc,
            runner_logloss=ll,
            top1_acc=t1,
            race_logloss=rll,
            model_best_iter=int(model.get_best_iteration() or -1),
        )
        all_window_metrics.append(metrics)

        scored = test[["race_date", "race_id", "horse_key", "y"]].copy()
        scored["p_win"] = p_test
        scored["window_test_start"] = str(w["test_start"])
        all_scored_rows.append(scored)

    metrics_df = pd.DataFrame(all_window_metrics)
    scored_df = pd.concat(all_scored_rows, ignore_index=True)

    # Overall summary (weighted by test races)
    weights = metrics_df["test_races"].astype(float).to_numpy()
    overall = pd.DataFrame([dict(
        windows=len(metrics_df),
        test_races_total=int(metrics_df["test_races"].sum()),
        runner_auc=float(np.average(metrics_df["runner_auc"], weights=weights)),
        runner_logloss=float(np.average(metrics_df["runner_logloss"], weights=weights)),
        top1_acc=float(np.average(metrics_df["top1_acc"], weights=weights)),
        race_logloss=float(np.average(metrics_df["race_logloss"], weights=weights)),
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        n_lag=N_LAG,
        features=len([c for c in build_feature_matrix(df)[3]]),
    )])

    # Write outputs
    to_csv_eu(metrics_df, OUTDIR / "restart_model_window_metrics.csv")
    to_csv_eu(overall, OUTDIR / "restart_model_overall.csv")
    to_csv_eu(scored_df, OUTDIR / "restart_model_scored_test.csv")

    print("\nWrote:")
    print(f" - {OUTDIR / 'restart_model_window_metrics.csv'}")
    print(f" - {OUTDIR / 'restart_model_overall.csv'}")
    print(f" - {OUTDIR / 'restart_model_scored_test.csv'}")

    print("\n=== OVERALL (weighted by test races) ===")
    print(overall.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
