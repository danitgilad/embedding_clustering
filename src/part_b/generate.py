"""Generate a face dataset from thispersondoesnotexist.com.

A plain HTTP GET returns one random JPEG. We loop to N, dedup by content hash (the site
can repeat images), and retry transient errors with backoff. `fetch` is injectable so
tests run without network.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Callable

from src.utils.io import ensure_dir

log = logging.getLogger(__name__)


def _default_fetch(url: str) -> bytes:
    """Fetch raw bytes from url with a browser-like UA (TPDNE rejects the default UA)."""
    import requests

    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    return resp.content


def generate_faces(n: int, url: str, out_dir: str | Path, delay_s: float,
                   max_retries: int, fetch: Callable[[str], bytes] | None = None) -> list[Path]:
    """Download n unique face JPEGs to out_dir. Returns saved paths (face_0000.jpg ...)."""
    fetch = fetch or _default_fetch
    out = ensure_dir(out_dir)
    seen: set[str] = set()
    saved: list[Path] = []
    attempts = 0
    max_attempts = n * (max_retries + 3) + 10
    while len(saved) < n and attempts < max_attempts:
        attempts += 1
        try:
            data = fetch(url)
        except Exception as exc:  # noqa: BLE001 - transient network; retry w/ backoff
            log.warning("fetch failed (%s); retrying", exc)
            time.sleep(min(2.0, delay_s + 0.2))
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            log.debug("duplicate image; skipping")
            continue
        seen.add(digest)
        p = out / f"face_{len(saved):04d}.jpg"
        p.write_bytes(data)
        saved.append(p)
        if delay_s:
            time.sleep(delay_s)
    log.info("Generated %d/%d unique faces (%d attempts)", len(saved), n, attempts)
    return saved
