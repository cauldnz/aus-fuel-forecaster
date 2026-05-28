"""Tests for fuel_pred.config — paths, constants, and the weather-source router.

The WEATHER_SOURCE router (spec §13.7 v2.0) decides which weather data
backend the pipeline uses: strict-free NOAA GFS via AWS S3, or the
paid-tier Open-Meteo API. The tests below pin down the resolution
semantics so a regression there can't silently route the fetcher to the
wrong source.
"""
from __future__ import annotations

import pytest

from fuel_pred import config

# ---------- resolve_weather_source ---------------------------------------


def test_resolve_weather_source_explicit_gfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """WEATHER_SOURCE=gfs always returns 'gfs' regardless of key presence."""
    monkeypatch.setenv("WEATHER_SOURCE", "gfs")
    monkeypatch.setenv("OPENMETEO_API_KEY", "some-key")
    assert config.resolve_weather_source() == "gfs"


def test_resolve_weather_source_explicit_openmeteo(monkeypatch: pytest.MonkeyPatch) -> None:
    """WEATHER_SOURCE=openmeteo always returns 'openmeteo', even without a key.

    (Without a key, the Open-Meteo fetcher will hit aggressive rate limits;
    that's a runtime concern surfaced by the fetcher, not a config-resolution
    concern. The router just picks the source the user asked for.)
    """
    monkeypatch.setenv("WEATHER_SOURCE", "openmeteo")
    monkeypatch.delenv("OPENMETEO_API_KEY", raising=False)
    assert config.resolve_weather_source() == "openmeteo"


def test_resolve_weather_source_auto_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto + OPENMETEO_API_KEY set → 'openmeteo' (paid-tier preferred when available)."""
    monkeypatch.setenv("WEATHER_SOURCE", "auto")
    monkeypatch.setenv("OPENMETEO_API_KEY", "test-key-value")
    assert config.resolve_weather_source() == "openmeteo"


def test_resolve_weather_source_auto_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto + no OPENMETEO_API_KEY → 'gfs' (strict-free default)."""
    monkeypatch.setenv("WEATHER_SOURCE", "auto")
    monkeypatch.delenv("OPENMETEO_API_KEY", raising=False)
    # Also clear the module-level value in case it was loaded from .env at import time.
    monkeypatch.setattr(config, "OPENMETEO_API_KEY", None)
    assert config.resolve_weather_source() == "gfs"


def test_resolve_weather_source_auto_with_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto + empty-string key → 'gfs' (empty string is treated as no key)."""
    monkeypatch.setenv("WEATHER_SOURCE", "auto")
    monkeypatch.setenv("OPENMETEO_API_KEY", "")
    monkeypatch.setattr(config, "OPENMETEO_API_KEY", None)
    assert config.resolve_weather_source() == "gfs"


def test_resolve_weather_source_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown WEATHER_SOURCE value raises with a helpful message."""
    monkeypatch.setenv("WEATHER_SOURCE", "lambda-from-azure")
    with pytest.raises(ValueError, match="Invalid WEATHER_SOURCE"):
        config.resolve_weather_source()


# ---------- path constants ------------------------------------------------


def test_path_constants_resolve_under_repo_root() -> None:
    """All DATA_* and the new GFS path constants live under REPO_ROOT."""
    assert config.DATA_DIR.is_relative_to(config.REPO_ROOT)
    assert config.RAW_WEATHER_GFS_DIR.is_relative_to(config.DATA_RAW)
    assert config.INTERIM_STATION_GRID_MAPPING.is_relative_to(config.DATA_INTERIM)


def test_raw_weather_gfs_dir_named_consistently_with_gfs_module() -> None:
    """The path constant must match what tools/parallel_gfs_fetch.py and
    src/fuel_pred/fetch/gfs.py write. If either gets renamed, this test
    will catch the inconsistency early."""
    assert config.RAW_WEATHER_GFS_DIR.name == "weather_gfs"
    assert config.INTERIM_STATION_GRID_MAPPING.name == "station_grid_mapping.parquet"
