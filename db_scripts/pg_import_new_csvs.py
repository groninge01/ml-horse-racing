#!/usr/bin/env python3
"""
pg_import_new_csvs.py

Incrementally import new race CSVs into runners_raw.

- Reads all *.csv files from ../tmp
- Uses COPY into the existing runners_raw table
- Relies on UNIQUE (race_id, horse, num) to avoid duplicates
- Safe to run multiple times

Usage:
  python pg_import_new_csvs.py
"""

from __future__ import annotations

from pathlib import Path
import psycopg


DSN = "postgresql://postgres:postgres@192.168.88.34:5432/postgres?sslmode=disable"
TABLE = "public.runners_raw"
CSV_DIR = Path("../tmp")


COPY_SQL = f"""
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


def main() -> None:
    csv_files = sorted(CSV_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {CSV_DIR.resolve()}")
        return

    print(f"Found {len(csv_files)} CSV files to import:")
    for f in csv_files:
        print(f"  - {f.name}")

    total_before = 0
    total_after = 0

    with psycopg.connect(DSN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
            total_before = cur.fetchone()[0]

            for csv_path in csv_files:
                print(f"\nImporting {csv_path.name} …")

                try:
                    with cur.copy(COPY_SQL) as copy:
                        with csv_path.open("rb") as f:
                            while chunk := f.read(1024 * 1024):
                                copy.write(chunk)

                    cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
                    after = cur.fetchone()[0]
                    inserted = after - total_before
                    total_before = after

                    print(f"  Rows added: {inserted:,}")

                except Exception as e:
                    print(f"  ERROR importing {csv_path.name}: {e}")

            total_after = total_before

    print("\n=== IMPORT SUMMARY ===")
    print(f"Total rows after import: {total_after:,}")
    print("Done.")


if __name__ == "__main__":
    main()
