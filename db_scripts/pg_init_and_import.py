#!/usr/bin/env python3
"""
pg_init_and_import.py

Creates runners_raw table using the EXACT raceform.csv schema
and imports historic data via COPY.

Usage:
  python pg_init_and_import.py data/raceform.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
import psycopg


DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
TABLE = "public.runners_raw"


# IMPORTANT:
# - Columns kept TEXT-first to guarantee load
# - SQL reserved words are quoted ("type", "class", "or", "time")
CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  date         DATE,
  course       TEXT,
  race_id      BIGINT,
  off          TEXT,
  race_name    TEXT,
  "type"       TEXT,
  "class"      TEXT,
  pattern      TEXT,
  rating_band TEXT,
  age_band     TEXT,
  sex_rest     TEXT,
  dist         TEXT,
  going        TEXT,
  ran          TEXT,
  num          TEXT,
  pos          TEXT,
  draw         TEXT,
  ovr_btn      TEXT,
  btn          TEXT,
  horse        TEXT,
  age          TEXT,
  sex          TEXT,
  wgt          TEXT,
  hg           TEXT,
  "time"       TEXT,
  sp           TEXT,
  jockey       TEXT,
  trainer      TEXT,
  prize        TEXT,
  "or"         TEXT,
  rpr          TEXT,
  ts           TEXT,
  sire         TEXT,
  dam          TEXT,
  damsire     TEXT,
  owner        TEXT,
  comment      TEXT,

  ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS runners_raw_race_id_idx ON {TABLE} (race_id);
CREATE INDEX IF NOT EXISTS runners_raw_date_idx    ON {TABLE} (date);
CREATE INDEX IF NOT EXISTS runners_raw_horse_idx   ON {TABLE} (horse);
"""

DEDUP_SQL = f"""
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'runners_raw_race_horse_num_uk'
  ) THEN
    ALTER TABLE {TABLE}
    ADD CONSTRAINT runners_raw_race_horse_num_uk
    UNIQUE (race_id, horse, num);
  END IF;
END $$;
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python pg_init_and_import.py <raceform.csv>")
        return 2

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}")
        return 2

    print("Connecting to Postgres…")
    with psycopg.connect(DSN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            print("Creating table…")
            cur.execute(CREATE_SQL)
            cur.execute(INDEX_SQL)
            cur.execute(DEDUP_SQL)

            # Uncomment ONLY if you want to reload everything
            # cur.execute(f"TRUNCATE TABLE {TABLE};")

            print(f"Importing CSV via COPY: {csv_path}")

            copy_sql = f"""
                COPY {TABLE} (
                  date, course, race_id, off, race_name,
                  "type", "class", pattern, rating_band, age_band,
                  sex_rest, dist, going, ran, num, pos,
                  draw, ovr_btn, btn, horse, age,
                  sex, wgt, hg, "time", sp,
                  jockey, trainer, prize, "or", rpr,
                  ts, sire, dam, damsire, owner, comment
                )
                FROM STDIN WITH (FORMAT csv, HEADER true)
            """

            with cur.copy(copy_sql) as copy:
                with csv_path.open("rb") as f:
                    while chunk := f.read(1024 * 1024):
                        copy.write(chunk)

            print("Import complete.")

            cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
            total = cur.fetchone()[0]
            print(f"Rows in runners_raw: {total:,}")

            cur.execute(f"SELECT MIN(date), MAX(date) FROM {TABLE};")
            mn, mx = cur.fetchone()
            print(f"Date range: {mn} → {mx}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
