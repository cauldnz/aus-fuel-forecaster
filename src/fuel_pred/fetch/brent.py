"""Fetch Brent crude futures daily OHLC via yfinance (ticker BZ=F).

Source: Yahoo Finance via the `yfinance` package.
Granularity: daily.
Coverage: 2000-01 → present.

Spec: spec.md §5.1.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(start: str, end: str, out: Path) -> None:
    """Fetch daily Brent OHLC and write Parquet with columns:
        date, open, high, low, close, volume
    """
    raise NotImplementedError("TODO: implement per spec.md §5.1.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    fetch(args.start, args.end, args.out)


if __name__ == "__main__":
    main()
