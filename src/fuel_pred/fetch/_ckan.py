"""Minimal CKAN client + shared HTTP helper used by the fetcher modules.

CKAN endpoints we depend on:
  - ``package_show?id=<package>`` → resource list
  - ``datastore_search?resource_id=<id>&offset=<n>&limit=<n>`` → paginated rows

Plus a generic ``download_bytes`` for resource downloads that aren't
served via the datastore.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from fuel_pred import config

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=30),
    reraise=True,
)
def download_bytes(url: str) -> bytes:
    """Fetch ``url`` and return the response body. Retries on transient errors."""
    logger.info("downloading %s", url)
    response = requests.get(
        url,
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.content


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=30),
    reraise=True,
)
def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    if not body.get("success", False):
        raise RuntimeError(f"CKAN error at {url}: {body.get('error')}")
    return body


def package_show(api_root: str, package_id: str) -> dict[str, Any]:
    """Return the ``result`` dict for a CKAN ``package_show`` call."""
    url = urljoin(api_root.rstrip("/") + "/", "package_show")
    body = _get_json(url, params={"id": package_id})
    result = body["result"]
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected package_show shape at {url}: {type(result).__name__}")
    return result


def iter_datastore(
    api_root: str,
    resource_id: str,
    *,
    page_size: int = 10000,
    log_every: int = 100_000,
) -> Iterator[list[dict[str, Any]]]:
    """Yield successive batches of records from a CKAN datastore resource.

    Pagination uses ``offset`` + ``limit``. We trust the server's ``total``
    field to bound the loop.
    """
    url = urljoin(api_root.rstrip("/") + "/", "datastore_search")
    offset = 0
    total: int | None = None
    fetched = 0
    last_logged = 0

    while True:
        body = _get_json(
            url,
            params={"resource_id": resource_id, "limit": page_size, "offset": offset},
        )
        result = body["result"]
        records = result.get("records", [])
        if total is None:
            total = int(result.get("total", 0))
            logger.info("datastore %s: total=%d", resource_id, total)

        if not records:
            break

        yield records
        fetched += len(records)

        if fetched - last_logged >= log_every:
            logger.info("datastore %s: %d / %d rows", resource_id, fetched, total)
            last_logged = fetched

        if fetched >= total:
            break

        offset += len(records)

    if total is not None and fetched != total:
        logger.warning(
            "datastore %s: fetched %d but total reported %d", resource_id, fetched, total
        )
