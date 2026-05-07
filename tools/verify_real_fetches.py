"""Real-network smoke test for the four Phase-1 fetchers.

Mirrors the `verify_real_parsers.py` pattern from `abs-census-augmentor`:
  - Hits real endpoints
  - Writes to a tempdir (does not touch `data/raw/`)
  - Prints row counts + a few sample rows for human eyeballing
  - Opt-in: not part of `pytest`/`make test`

Usage:

    uv run python tools/verify_real_fetches.py            # all four
    uv run python tools/verify_real_fetches.py brent      # one
    uv run python tools/verify_real_fetches.py brent audusd

Exit code: 0 if every requested fetcher returned non-empty output, 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fuel_pred.fetch import audusd, brent, fuelcheck, traffic

logger = logging.getLogger("verify_real_fetches")

# A small date window so this stays a smoke test, not a full backfill.
SMOKE_START: str = "2024-08-01"
SMOKE_END: str = "2024-08-31"


def _print_sample(name: str, df: pd.DataFrame, limit: int = 3) -> None:
    print(f"\n=== {name}: {len(df):,} rows, {len(df.columns)} cols ===")
    print("columns:", list(df.columns))
    if len(df) == 0:
        print("(empty)")
        return
    print(df.head(limit).to_string(index=False))


def verify_brent(out_dir: Path) -> bool:
    out = out_dir / "brent.parquet"
    brent.fetch(SMOKE_START, SMOKE_END, out, force=True)
    df = pd.read_parquet(out)
    _print_sample("brent", df)
    return len(df) > 0


def verify_audusd(out_dir: Path) -> bool:
    out = out_dir / "audusd.parquet"
    audusd.fetch(SMOKE_START, SMOKE_END, out, force=True)
    df = pd.read_parquet(out)
    _print_sample("audusd", df)
    return len(df) > 0


def verify_traffic(out_dir: Path) -> bool:
    target = out_dir / "traffic"
    traffic.fetch(SMOKE_START, SMOKE_END, target, force=True)
    stations = pd.read_parquet(target / "stations.parquet")
    hourly = pd.read_parquet(target / "hourly.parquet")
    _print_sample("traffic stations", stations)
    _print_sample("traffic hourly", hourly)
    return len(stations) > 0


def verify_fuelcheck(out_dir: Path) -> bool:
    target = out_dir / "fuelcheck"
    fuelcheck.fetch(SMOKE_START, SMOKE_END, target, force=True)
    files = sorted(target.glob("*.parquet"))
    print(f"\n=== fuelcheck: {len(files)} monthly parquet(s) ===")
    if not files:
        return False
    for path in files:
        df = pd.read_parquet(path)
        _print_sample(f"fuelcheck {path.stem}", df)
    return True


VERIFIERS: dict[str, Callable[[Path], bool]] = {
    "brent": verify_brent,
    "audusd": verify_audusd,
    "traffic": verify_traffic,
    "fuelcheck": verify_fuelcheck,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fetchers",
        nargs="*",
        choices=list(VERIFIERS.keys()),
        help="Subset to run (default: all)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    chosen = args.fetchers or list(VERIFIERS.keys())

    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="verify_real_fetches_") as tmp:
        out_dir = Path(tmp)
        for name in chosen:
            print(f"\n>>> running {name} ...")
            try:
                ok = VERIFIERS[name](out_dir)
            except Exception:
                traceback.print_exc()
                ok = False
            if not ok:
                failed.append(name)

    print("\n=== summary ===")
    for name in chosen:
        status = "FAIL" if name in failed else "OK"
        print(f"  {name}: {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
