"""Shared parser for RBA F-/G-series statistical CSVs and XLS files.

All RBA tables follow the same layout:

  - Multi-row text preamble (Title / Description / Frequency / Type /
    Units / Source / Publication date), variable line count.
  - One ``Series ID`` row whose cells (after the first) are the
    column-level mnemonic codes (e.g. ``FXRUSD``, ``FIRMMCRTD``).
  - Data rows below: column 0 is a date, the remaining columns are
    one numeric value per series.

The CSV form has a ragged shape (the title row is one cell wide; data
rows are many) which both pandas parsers refuse — we read with stdlib
csv and right-pad. The XLS form is read via ``openpyxl``/``xlrd``.

This module exists so ``fetch.audusd``, ``fetch.cash_rate``, and
``fetch.inflation_expectations`` can share the parsing surface
without copy-paste.
"""
from __future__ import annotations

import csv
import io

import pandas as pd


def read_rba_table(payload: bytes, url: str) -> pd.DataFrame:
    """Read an RBA F-/G-series payload into a header-less DataFrame.

    Picks the right reader (CSV stdlib or pandas read_excel) based on
    the URL extension. Returns a string-typed frame with positional
    columns; the caller picks rows and columns by content.
    """
    if url.lower().endswith(".csv"):
        text = payload.decode("utf-8-sig")
        # The RBA CSV has a single-cell title row followed by metadata rows
        # and data rows with many columns. pandas' parsers (both engines)
        # struggle with the variable column count, so we read with stdlib
        # csv and right-pad each row to a uniform width.
        rows = list(csv.reader(io.StringIO(text)))
        width = max((len(r) for r in rows), default=0)
        padded = [r + [""] * (width - len(r)) for r in rows]
        return pd.DataFrame(padded, dtype=str)
    if url.lower().endswith(".xlsx"):
        return pd.read_excel(
            io.BytesIO(payload),
            header=None,
            dtype=str,
            keep_default_na=False,
            engine="openpyxl",
        )
    return pd.read_excel(
        io.BytesIO(payload), header=None, dtype=str, keep_default_na=False, engine="xlrd"
    )


def parse_rba_table(
    payload: bytes,
    url: str,
    series_id: str,
    *,
    value_column_name: str,
) -> pd.DataFrame:
    """Extract a (date, <value_column_name>) frame for one series.

    Args:
        payload: raw bytes of the RBA file.
        url: the source URL — used to choose CSV vs XLS reader and for
            error messages.
        series_id: the RBA mnemonic (e.g. ``FXRUSD`` for AUD/USD,
            ``FIRMMCRTD`` for cash-rate target, ``GCONEXP`` for
            consumer inflation expectations).
        value_column_name: the name to give the value column in the
            returned DataFrame (e.g. ``"audusd"``, ``"cash_rate"``).

    Returns:
        DataFrame with columns ``date`` (datetime.date) and
        ``<value_column_name>`` (float). Rows where either is null are
        dropped.
    """
    raw = read_rba_table(payload, url)

    # Find the "Series ID" header row.
    first_col = raw.iloc[:, 0].astype(str).str.strip().str.casefold()
    matches = first_col.index[first_col == "series id"].tolist()
    if not matches:
        raise RuntimeError(f"could not find 'Series ID' row in {url}")
    series_row: int = int(matches[0])

    # Locate the column index that holds our series.
    series_row_values = raw.iloc[series_row].astype(str).str.strip()
    value_col: int = -1
    for col_idx in range(raw.shape[1]):
        if str(series_row_values.iloc[col_idx]) == series_id:
            value_col = col_idx
            break
    if value_col < 0:
        raise RuntimeError(f"series {series_id!r} not present in {url}")

    body = raw.iloc[series_row + 1 :].copy()
    date_series = pd.to_datetime(body.iloc[:, 0], errors="coerce", dayfirst=True)
    value_series = pd.to_numeric(body.iloc[:, value_col], errors="coerce")

    out = pd.DataFrame({"date": date_series, value_column_name: value_series})
    out = out[out["date"].notna() & out[value_column_name].notna()].copy()
    out["date"] = out["date"].dt.date

    return out.reset_index(drop=True)
