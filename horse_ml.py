#!/usr/bin/env python3
"""
horse_ml.py

Horse racing ML (PURE FORM) with market evaluation.

TRAINING (no leakage):
- Predict WIN (pos == 1)
- Time-based split (80/20)
- Uses only pre-race / form features
- Explicitly EXCLUDES post-race leakage fields like:
  pos/time/btn/ovr_btn/comment/rpr/ts and any odds-derived features

EVALUATION ONLY:
- Parses SP to decimal for benchmarking and value-betting simulation
- Extracts Opening Price (OP) from comment (e.g. "op 7/2", "op 1/2")
- Runs value betting sims vs OP and vs SP (optional odds caps)

OUTPUT:
- Writes ALL CSVs to ./output in EU formatting:
    sep=';'
    decimal=','
    date_format='DD-MM-YYYY'
"""

import sys
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve


# =============================
# Parsing helpers
# =============================

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def parse_weight_lbs(w: str) -> Optional[float]:
    """
    Weight like '11-6' => 11*14+6 lbs
    """
    if w is None or (isinstance(w, float) and np.isnan(w)):
        return None
    w = normalize_text(w)
    if w in ("", "-", "—", "–"):
        return None
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", w)
    if not m:
        return None
    return int(m.group(1)) * 14 + int(m.group(2))

def parse_fractional_to_decimal(frac: str) -> Optional[float]:
    """
    Fractional odds '7/2' -> decimal 4.5
    Returns decimal odds inclusive of stake (1 + num/den)
    """
    if frac is None or (isinstance(frac, float) and np.isnan(frac)):
        return None
    s = normalize_text(frac).upper()
    s = s.replace("EVENS", "EVS")
    # strip trailing letters (F, J, etc.)
    s = re.sub(r"[A-Z]+$", "", s).strip()
    if s in ("", "-", "—", "–"):
        return None
    if s in ("EVS", "EV"):
        return 2.0
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return None
    return 1.0 + (num / den)

def parse_sp_to_decimal(sp: str) -> Optional[float]:
    """
    SP formats like '25/1', '8/15F', '1/3F', 'Evs'
    """
    if sp is None or (isinstance(sp, float) and np.isnan(sp)):
        return None
    s = normalize_text(sp).upper()
    s = s.replace("EVENS", "EVS")
    s = re.sub(r"[A-Z]+$", "", s).strip()
    return parse_fractional_to_decimal(s)

def dist_to_furlongs(dist: str) -> Optional[float]:
    """
    Parse distance strings like '2m3½f', '2m5f', '2m', '1m7f'
    1 mile = 8 furlongs
    """
    if dist is None or (isinstance(dist, float) and np.isnan(dist)):
        return None
    s = normalize_text(dist)
    s = s.replace("Â½", "½").replace("â½", "½")

    miles = 0.0
    furlongs = 0.0

    mm = re.search(r"(\d+)\s*m", s)
    if mm:
        miles = float(mm.group(1))

    mf = re.search(r"(\d+)(½)?\s*f", s)
    if mf:
        furlongs = float(mf.group(1)) + (0.5 if mf.group(2) else 0.0)

    return miles * 8.0 + furlongs

def pos_is_win(pos) -> Optional[int]:
    """
    pos numeric => 1 means win else 0
    non-finish codes like PU/UR => 0
    """
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return None
    s = normalize_text(pos)
    if s == "":
        return None
    try:
        return 1 if int(float(s)) == 1 else 0
    except Exception:
        return 0

def extract_op_frac_from_comment(comment: str) -> Optional[str]:
    """
    Extract opening odds from Racing Post style comment:
      "... (op 7/2) ..." OR "... (op 1/2) ..."
    If 'tchd X/Y' exists without 'op', return None (or fall back logic if you want).
    We'll extract 'op' specifically.
    """
    if comment is None or (isinstance(comment, float) and np.isnan(comment)):
        return None
    c = normalize_text(comment).lower()

    # Most common: "(op 7/2)" or "op 7/2"
    m = re.search(r"\bop\s+(\d+\s*/\s*\d+)\b", c)
    if m:
        return m.group(1).replace(" ", "")

    # Sometimes could be "op7/2" (rare) - handle it
    m2 = re.search(r"\bop(\d+\s*/\s*\d+)\b", c)
    if m2:
        return m2.group(1).replace(" ", "")

    # If you want to treat 'tchd' as "no OP printed" and fall back to SP, do it later in eval
    return None


# =============================
# EU CSV export
# =============================

def to_csv_eu(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        sep=";",
        decimal=",",
        date_format="%d-%m-%Y",
    )


# =============================
# Betting simulation
# =============================

def simulate_value_bets(
    df: pd.DataFrame,
    p_col: str,
    odds_dec_col: str,
    margin: float,
    stake: float = 1.0,
    max_odds: Optional[float] = 12.0,
) -> Dict[str, Any]:
    """
    Bet if model probability exceeds implied probability by margin:
      implied p = 1 / odds_dec
      bet if p_model > implied_p * (1 + margin)

    Profit:
      win: +(odds_dec - 1) * stake
      lose: -stake

    max_odds: optional cap (e.g. 12.0 means <= 11/1) to avoid longshot bias.
    """
    d = df.dropna(subset=[p_col, odds_dec_col, "y"]).copy()
    if d.empty:
        return {"bets": 0, "strike": 0.0, "roi": 0.0, "avg_odds": np.nan, "profit": 0.0}

    d[p_col] = d[p_col].astype(float)
    d[odds_dec_col] = d[odds_dec_col].astype(float)

    if max_odds is not None:
        d = d[d[odds_dec_col] <= float(max_odds)]
        if d.empty:
            return {"bets": 0, "strike": 0.0, "roi": 0.0, "avg_odds": np.nan, "profit": 0.0}

    implied = 1.0 / d[odds_dec_col]
    bets = d[d[p_col] > implied * (1.0 + margin)].copy()
    if bets.empty:
        return {"bets": 0, "strike": 0.0, "roi": 0.0, "avg_odds": np.nan, "profit": 0.0}

    wins = bets["y"].astype(int) == 1
    profit = np.where(wins, (bets[odds_dec_col] - 1.0) * stake, -stake).sum()
    n = len(bets)
    roi = profit / (n * stake)
    strike = float(wins.mean())
    avg_odds = float(bets[odds_dec_col].mean())

    return {"bets": int(n), "strike": strike, "roi": float(roi), "avg_odds": avg_odds, "profit": float(profit)}


# =============================
# Main
# =============================

def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rank-in-race features to avoid leaking via race_id but still model field context.
    Uses 'race_id' only for grouping, NOT as a model feature.
    """
    out = df.copy()
    if "race_id" not in out.columns:
        return out

    # Field size: prefer 'ran' if present, else compute from group size
    if "ran" not in out.columns:
        out["ran"] = out.groupby("race_id")["race_id"].transform("size")

    # Helper to compute rank with safe handling
    def rank_col(col: str, ascending: bool = True, name: str = ""):
        if col not in out.columns:
            return
        out[name] = out.groupby("race_id")[col].rank(method="average", ascending=ascending)

    # Lower OR rank number means "lower value"; but higher OR is better.
    # We'll rank descending for "best is 1".
    if "or" in out.columns:
        rank_col("or", ascending=False, name="or_rank")

    # Lower weight often advantageous; rank ascending so "lowest weight is 1"
    if "wgt_lbs" in out.columns:
        rank_col("wgt_lbs", ascending=True, name="wgt_rank")

    # Age: depends by race type; but rank can still capture relative profile.
    if "age" in out.columns:
        rank_col("age", ascending=True, name="age_rank")

    # Draw: depends; include rank anyway for context
    if "draw" in out.columns:
        rank_col("draw", ascending=True, name="draw_rank")

    return out

def main(csv_path: str):
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_safe(csv_path)
    df.columns = [normalize_text(c) for c in df.columns]

    # Minimal required
    for req in ("date", "race_id", "pos"):
        if req not in df.columns:
            raise SystemExit(f"CSV must contain '{req}' column. Found: {list(df.columns)[:50]}...")

    # Date parsing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Target
    df["y"] = df["pos"].apply(pos_is_win)
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    # Parse pre-race-ish structured features
    if "wgt" in df.columns:
        df["wgt_lbs"] = df["wgt"].apply(parse_weight_lbs)
    else:
        df["wgt_lbs"] = np.nan

    if "dist" in df.columns:
        df["dist_f"] = df["dist"].apply(dist_to_furlongs)
    else:
        df["dist_f"] = np.nan

    # Odds parsing for evaluation only
    if "sp" in df.columns:
        df["sp_dec"] = df["sp"].apply(parse_sp_to_decimal)
    else:
        df["sp_dec"] = np.nan

    if "comment" in df.columns:
        df["op_frac"] = df["comment"].apply(extract_op_frac_from_comment)
        df["op_dec"] = df["op_frac"].apply(parse_fractional_to_decimal)
    else:
        df["op_frac"] = np.nan
        df["op_dec"] = np.nan

    # Numeric coercions (pre-race)
    for col in ["ran", "draw", "age", "or"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Explicitly DROP/IGNORE known post-race leakage columns (if present)
    # We won't train on them regardless; this is just defensive documentation.
    leakage_cols = [c for c in ["rpr", "ts", "time", "btn", "ovr_btn", "prize"] if c in df.columns]
    # (We won't drop them globally because they may be useful for analysis later,
    #  but we will not include them in training features.)

    # Require key pre-race numeric basics
    df = df.dropna(subset=["dist_f", "wgt_lbs"], how="any")

    # Add rank-in-race features (uses race_id only for grouping)
    df = add_rank_features(df)

    # -----------------------------
    # TRAINING FEATURES (PURE FORM)
    # -----------------------------
    # IMPORTANT:
    # - Do NOT include: race_id, sp*, op*, rpr, ts, time, btn, ovr_btn, comment
    # - Keep: or (pre-race), distance/going/class, age/sex, weight, draw, jockey/trainer, pedigree, ranks
    candidates = [
        "course",
        "off",
        "race_name",
        "type",
        "class",
        "pattern",
        "rating_band",
        "age_band",
        "sex_rest",
        "dist_f",
        "going",
        "ran",
        "draw",
        "age",
        "sex",
        "wgt_lbs",
        "jockey",
        "trainer",
        "owner",
        "or",
        "sire",
        "dam",
        "damsire",
        # rank-in-race engineered
        "or_rank",
        "wgt_rank",
        "age_rank",
        "draw_rank",
    ]
    features = [c for c in candidates if c in df.columns]

    # Ensure we are NOT accidentally including leakage
    banned_prefixes = ("sp_", "op_")
    banned_exact = {"race_id", "sp", "comment", "rpr", "ts", "time", "btn", "ovr_btn", "pos"}
    features = [
        f for f in features
        if f not in banned_exact and not f.startswith(banned_prefixes)
    ]

    if not features:
        raise SystemExit("No usable training features found after filtering.")

    # CatBoost: fill NaN in categorical features with string; numeric NaNs can remain
    for col in features:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("UNKNOWN")

    # Time split
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train, y_train = train[features], train["y"]
    X_test, y_test = test[features], test["y"]

    cat_cols = [c for c in features if X_train[c].dtype == "object"]
    cat_idx = [features.index(c) for c in cat_cols]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=2000,
        depth=8,
        learning_rate=0.06,
        random_seed=42,
        verbose=200,
        thread_count=-1,
        od_type="Iter",
        od_wait=80,
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    p_test = model.predict_proba(test_pool)[:, 1]
    test["p_win_form"] = p_test

    # -----------------------------
    # Metrics (model + baselines)
    # -----------------------------
    auc = roc_auc_score(y_test, p_test)
    ll = log_loss(y_test, p_test)

    # SP baseline where available
    sp_mask = test["sp_dec"].notna()
    sp_auc = np.nan
    sp_ll = np.nan
    if sp_mask.any():
        sp_p = (1.0 / test.loc[sp_mask, "sp_dec"].astype(float)).clip(1e-6, 1 - 1e-6)
        sp_auc = float(roc_auc_score(test.loc[sp_mask, "y"], sp_p))
        sp_ll = float(log_loss(test.loc[sp_mask, "y"], sp_p))

    # OP baseline where available
    op_mask = test["op_dec"].notna()
    op_auc = np.nan
    op_ll = np.nan
    if op_mask.any():
        op_p = (1.0 / test.loc[op_mask, "op_dec"].astype(float)).clip(1e-6, 1 - 1e-6)
        op_auc = float(roc_auc_score(test.loc[op_mask, "y"], op_p))
        op_ll = float(log_loss(test.loc[op_mask, "y"], op_p))

    print("\n=== PURE FORM MODEL METRICS (OOS) ===")
    print(f"Rows train: {len(train):,} | test: {len(test):,}")
    print(f"AUC:     {auc:.4f}")
    print(f"LogLoss: {ll:.4f}")

    print("\n=== IMPLIED ODDS BASELINES (OOS) ===")
    if sp_mask.any():
        print(f"SP AUC:     {sp_auc:.4f}")
        print(f"SP LogLoss: {sp_ll:.4f}")
    else:
        print("SP baseline: no usable SP rows in test set.")

    if op_mask.any():
        print(f"OP AUC:     {op_auc:.4f}")
        print(f"OP LogLoss: {op_ll:.4f}")
    else:
        print("OP baseline: no usable OP rows in test set.")

    print("\n=== CALIBRATION (pure form model) ===")
    frac, mean = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    for i, (m, f) in enumerate(zip(mean, frac), 1):
        print(f"Bin {i:02d}: pred={m:.3f} actual={f:.3f}")

    # -----------------------------
    # Value betting sims (realistic caps)
    # -----------------------------
    # Odds caps to reduce longshot-bias artifacts.
    # Default: <= 12.0 (11/1). You can change these.
    MAX_ODDS_SP = 12.0
    MAX_ODDS_OP = 12.0

    print("\n=== VALUE BETS (pure form model) ===")
    if sp_mask.any():
        eval_sp = test.loc[sp_mask].copy()
        print(f"vs SP (cap <= {MAX_ODDS_SP}):")
        for margin in (0.02, 0.05, 0.08, 0.10):
            r = simulate_value_bets(eval_sp, "p_win_form", "sp_dec", margin=margin, max_odds=MAX_ODDS_SP)
            print(f"  Edge {margin:0.2f}: bets={r['bets']:6d} strike={r['strike']:.3f} ROI={r['roi']:.3f} avg_odds={r['avg_odds']:.2f}")
    else:
        print("vs SP: no usable SP rows.")

    if op_mask.any():
        eval_op = test.loc[op_mask].copy()
        print(f"vs OP (cap <= {MAX_ODDS_OP}):")
        for margin in (0.02, 0.05, 0.08, 0.10):
            r = simulate_value_bets(eval_op, "p_win_form", "op_dec", margin=margin, max_odds=MAX_ODDS_OP)
            print(f"  Edge {margin:0.2f}: bets={r['bets']:6d} strike={r['strike']:.3f} ROI={r['roi']:.3f} avg_odds={r['avg_odds']:.2f}")
    else:
        print("vs OP: no usable OP rows.")

    # -----------------------------
    # Feature importance
    # -----------------------------
    fi = model.get_feature_importance(train_pool)
    fi_df = (
        pd.DataFrame({"feature": features, "importance": fi})
        .sort_values("importance", ascending=False)
    )
    to_csv_eu(fi_df, out_dir / "horse_ml_feature_importance.csv")
    print("\nWrote: output/horse_ml_feature_importance.csv")

    # -----------------------------
    # Save scored test + metrics
    # -----------------------------
    # Add implied probs for comparison where available
    scored = test.copy()
    scored["sp_implied"] = np.where(scored["sp_dec"].notna(), 1.0 / scored["sp_dec"].astype(float), np.nan)
    scored["op_implied"] = np.where(scored["op_dec"].notna(), 1.0 / scored["op_dec"].astype(float), np.nan)

    out_cols = [
        "date", "course", "race_id", "off", "race_name", "horse",
        "dist", "dist_f", "going", "ran", "draw", "age", "sex", "wgt", "wgt_lbs",
        "jockey", "trainer", "owner",
        "or",
        "or_rank", "wgt_rank", "age_rank", "draw_rank",
        "sp", "sp_dec", "sp_implied",
        "op_frac", "op_dec", "op_implied",
        "p_win_form", "y",
    ]
    out_cols = [c for c in out_cols if c in scored.columns]

    to_csv_eu(scored[out_cols], out_dir / "horse_ml_scored_test.csv")
    print("Wrote: output/horse_ml_scored_test.csv")

    metrics = [{
        "train_rows": len(train),
        "test_rows": len(test),
        "model_auc": auc,
        "model_logloss": ll,
        "sp_rows": int(sp_mask.sum()),
        "sp_auc": sp_auc,
        "sp_logloss": sp_ll,
        "op_rows": int(op_mask.sum()),
        "op_auc": op_auc,
        "op_logloss": op_ll,
        "max_odds_sp": MAX_ODDS_SP,
        "max_odds_op": MAX_ODDS_OP,
        "leakage_cols_present": ",".join(leakage_cols) if leakage_cols else "",
    }]
    to_csv_eu(pd.DataFrame(metrics), out_dir / "horse_ml_metrics.csv")
    print("Wrote: output/horse_ml_metrics.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python horse_ml.py data/raceform.csv")
    main(sys.argv[1])
