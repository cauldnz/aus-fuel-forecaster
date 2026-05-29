"""Incremental backup of `data/raw/weather/*.parquet` to a destination dir.

The weather cache is gitignored (large, regenerable in principle, but expensive
in API quota — yesterday burned through Open-Meteo's 10,000/day cap re-fetching).
This script mirrors the cache to a chosen destination — typically a OneDrive
folder, which then auto-syncs to cloud — so a disk failure or accidental delete
doesn't cost us another day of fetching.

Two usage patterns:

1. **One-shot mirror after a fetch completes:**
   ```
   uv run python tools/sync_weather_cache.py \\
       --dest "C:/Users/<you>/OneDrive/fuel-pred-backups/weather"
   ```

2. **Background watcher while a fetch is running** (`--watch` mode):
   ```
   uv run python tools/sync_weather_cache.py \\
       --dest "..." --watch --interval 60
   ```
   Polls every N seconds, copies any newly-arrived parquets. Cheap when the
   destination already has the file (skipped via mtime+size comparison).

Both modes skip the source's `.tmp` files (the in-progress writes from the
atomic-write pattern in `fetch.weather.fetch_one`).

Notes:
- The destination can be on a different drive; copy is plain `shutil.copy2`
  so file metadata (mtime, size) is preserved.
- This is a `tools/` script per project convention — not a pipeline module.
- Safe to interrupt at any time; the next run picks up where it left off.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_weather_cache")


def _needs_copy(src: Path, dst: Path) -> bool:
    """True if dst doesn't exist OR is older/smaller than src.

    Cheap proxy for content equality. We could hash both but for the
    expected volume (~4500 small parquets, ~35KB each) and the fact
    that mtime+size matches are extremely rare for genuinely different
    content, the proxy is fine.
    """
    if not dst.exists():
        return True
    src_stat = src.stat()
    dst_stat = dst.stat()
    if src_stat.st_size != dst_stat.st_size:
        return True
    # Allow 1s of slack to absorb FS timestamp resolution differences.
    return src_stat.st_mtime > dst_stat.st_mtime + 1.0


def sync_once(src_dir: Path, dst_dir: Path) -> tuple[int, int, int]:
    """Mirror src_dir → dst_dir. Returns (copied, skipped, errors)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    errors = 0
    for src in src_dir.glob("*.parquet"):
        # Skip in-flight atomic writes
        if src.suffix == ".tmp" or src.name.endswith(".parquet.tmp"):
            continue
        dst = dst_dir / src.name
        if not _needs_copy(src, dst):
            skipped += 1
            continue
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as exc:
            errors += 1
            logger.warning("FAIL copy %s -> %s: %s", src, dst, exc)
    return copied, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, default=Path("data/raw/weather"),
        help="Source weather cache directory (default: data/raw/weather)",
    )
    parser.add_argument(
        "--dest", type=Path, required=True,
        help="Destination dir (typically a OneDrive folder for cloud sync)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Poll forever; copy new files as they arrive. Stop with Ctrl-C.",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Watch poll interval in seconds (default 60). Only used with --watch.",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source dir {args.src} does not exist or is not a directory")

    if args.watch:
        logger.info(
            "watching %s -> %s every %ds (Ctrl-C to stop)",
            args.src, args.dest, args.interval,
        )
        try:
            while True:
                copied, skipped, errors = sync_once(args.src, args.dest)
                if copied or errors:
                    logger.info(
                        "sync: copied=%d skipped=%d errors=%d",
                        copied, skipped, errors,
                    )
                else:
                    logger.debug("sync: no new files (skipped=%d)", skipped)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("watcher stopped by user")
    else:
        copied, skipped, errors = sync_once(args.src, args.dest)
        logger.info(
            "one-shot sync complete: copied=%d skipped=%d errors=%d",
            copied, skipped, errors,
        )


if __name__ == "__main__":
    main()
