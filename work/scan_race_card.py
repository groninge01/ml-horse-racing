#!/usr/bin/env python3
import sys
import json
import pandas as pd
from pathlib import Path

CSV_SEP = ";"
EDGE_THRESHOLD = 0.03
TOP_K = 1

REQUIRED_MODEL_COLS = {
    "race_id",
    "horse_key",
    "p_win",
}

# =========================
# LOAD MODEL SCORES
# =========================
def load_model_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=CSV_SEP)

    missing = REQUIRED_MODEL_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing model columns: {missing}")

    df["p_model"] = (
        df["p_win"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["horse_key"] = df["horse_key"].astype(str)
    df["race_id"] = df["race_id"].astype(int)

    return df[["race_id", "horse_key", "p_model"]]


# =========================
# PARSE RACE CARD
# =========================
def load_racecard(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 3:
        print("Usage: python scan_race_card_chase.py <racecard.json> <scored_model.csv>")
        sys.exit(1)

    racecard_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])

    card = load_racecard(racecard_path)
    model = load_model_scores(model_path)

    found_any = False

    for region in card.values():
        for meeting in region.values():
            for race in meeting.values():

                # ✅ CORRECT CHECK
                if race.get("race_type") != "Chase":
                    continue

                race_id = race["race_id"]
                runners = race["runners"]

                race_df = pd.DataFrame({
                    "horse_key": [r["name"] for r in runners],
                })

                race_df["race_id"] = race_id

                merged = race_df.merge(
                    model,
                    on=["race_id", "horse_key"],
                    how="left"
                ).dropna(subset=["p_model"])

                if merged.empty:
                    continue

                merged["model_rank"] = merged["p_model"].rank(
                    ascending=False,
                    method="first"
                )

                merged["edge"] = merged["p_model"] - merged["p_model"].mean()

                picks = merged[
                    (merged["model_rank"] <= TOP_K) &
                    (merged["edge"] >= EDGE_THRESHOLD)
                ]

                if picks.empty:
                    continue

                found_any = True

                print("\n==============================")
                print(f"CHASE RACE FOUND: {race['off_time']} {race['course']}")
                print(f"Race ID: {race_id}")
                print("==============================")

                print(
                    picks[[
                        "horse_key",
                        "p_model",
                        "edge",
                        "model_rank"
                    ]].sort_values("model_rank")
                )

    if not found_any:
        print("No Chase races with qualifying model selections on this card.")


if __name__ == "__main__":
    main()
