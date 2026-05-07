"""Fetch ANZ-Roy Morgan Consumer Confidence weekly index.

Source: https://www.roymorgan.com/findings/anz-roy-morgan-consumer-confidence
Granularity: weekly (Friday release).
Coverage: late 1990s → present.

Granularity note: Roy Morgan does not publish a clean machine-readable feed.
This fetcher likely needs to scrape an HTML table or download a published
PDF/Excel. Be conservative; cache aggressively. The feature builder
forward-fills weekly values to daily.

Spec: spec.md §5.2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(start: str, end: str, out: Path) -> None:
    """Fetch ANZ-RM Consumer Confidence weekly, write Parquet:
        date, consumer_confidence
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
