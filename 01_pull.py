"""Download nflverse data once and save it locally as parquet (raw layer).

Past seasons are only downloaded if missing. The current season (2026) is
always re-downloaded because nflverse updates it nightly.

Usage:
    .venv/bin/python 01_pull.py            # pull anything missing + refresh 2026
    .venv/bin/python 01_pull.py --force    # re-download everything
"""

import sys
import time
from pathlib import Path

import nflreadpy as nfl
import polars as pl

FIRST_SEASON = 1999
CURRENT_SEASON = 2026
RAW = Path(__file__).parent / "data" / "raw"
FORCE = "--force" in sys.argv


def save(df: pl.DataFrame, name: str) -> None:
    path = RAW / f"{name}.parquet"
    df.write_parquet(path)
    print(f"  saved {path.name}: {df.height:,} rows x {df.width} cols")


def pull_season(season: int, retries: int = 3) -> None:
    path = RAW / f"pbp_{season}.parquet"
    if path.exists() and season != CURRENT_SEASON and not FORCE:
        print(f"pbp {season}: already have it, skipping")
        return
    for attempt in range(1, retries + 1):
        try:
            print(f"pbp {season}: downloading")
            save(nfl.load_pbp(seasons=[season]), f"pbp_{season}")
            return
        except Exception as e:  # network hiccups are common, retry
            print(f"  attempt {attempt} failed: {e}")
            time.sleep(3 * attempt)
    print(f"  !! gave up on {season}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        pull_season(season)

    # Small supporting tables: always refresh (cheap).
    print("schedules")
    save(nfl.load_schedules(seasons=True), "schedules")  # spread, roof, temp, wind, coaches, scores

    print("injuries (2009+)")
    try:
        save(nfl.load_injuries(seasons=list(range(2009, CURRENT_SEASON + 1))), "injuries")
    except Exception as e:
        # The newest season sometimes isn't published yet; fall back to completed seasons.
        print(f"  full range failed ({e}); trying up to {CURRENT_SEASON - 1}")
        save(nfl.load_injuries(seasons=list(range(2009, CURRENT_SEASON))), "injuries")


if __name__ == "__main__":
    main()
