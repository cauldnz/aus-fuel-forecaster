"""Fetch ASX 200 daily close via yfinance (ticker ^AXJO).

Source: Yahoo Finance.
Granularity: daily, business days.
Coverage: 1990s → present.

Spec: spec.md §5.2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(start: str, end: str, out: Path) -> None:
    """Fetch ASX 200 daily close, write Parquet:
        date, asx200_close
    """
    raise NotImplementedError("TODO: implement per spec.md §5.2.")


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
