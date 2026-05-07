"""Fetch RBA cash rate target daily series from RBA F1.1.

Source: https://www.rba.gov.au/statistics/historical-data.html#interest-rates
        (F1.1 historical cash rate target)
Granularity: forward-filled to daily (changes only on RBA decision dates).
Coverage: 1990 → present.

Spec: spec.md §5.2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(start: str, end: str, out: Path) -> None:
    """Fetch RBA cash rate, forward-fill to daily, write Parquet:
        date, cash_rate
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
