"""Dimensions normalization and run hashing.

Pure helpers with no dependency on run lifecycle state: cleaning arbitrary
params into BSON/JSON-safe dimensions, hashing them into a stable dedup
key, and looking that key up in either backend. :mod:`tracking.experiment`
owns the run lifecycle and imports from here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tracking.configurable import Configurable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BSON-safe cleaning (replaces the old ``clean_value``)
# ---------------------------------------------------------------------------

_TRUNCATE_AT = 4000


def dims_of(v: Any) -> dict[str, Any] | None:
    """Return ``v.dims()`` if it defines that protocol, else ``None``.

    The ``dims()`` protocol lets a parameter object control how it is
    tracked: instead of falling through to ``str(v)`` (a verbose blob for
    dataclasses carrying large payloads like geometries), the returned
    dict of plain scalar values is stored. Only a ``dict`` return value
    is honored; anything else falls through to the normal handling.
    """
    dims = getattr(v, "dims", None)
    if callable(dims):
        d = dims()
        if isinstance(d, dict):
            return d
    return None


def to_bsonable(v: Any) -> Any:
    """Convert arbitrary benchmark params/metrics into BSON-encodable values."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.number):
        return v.item()
    if isinstance(v, np.ndarray):
        return [to_bsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [to_bsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_bsonable(x) for k, x in v.items()}
    if isinstance(v, (Path,)):
        return str(v)
    d = dims_of(v)
    if d is not None:
        return {str(k): to_bsonable(x) for k, x in d.items()}
    if inspect.isfunction(v) or inspect.isbuiltin(v):
        # Stable across processes: str() would embed a memory address,
        # making stored configs (and hashes over them) unmatchable.
        return v.__name__
    if isinstance(v, Configurable):
        return v.__class__.__name__
    if hasattr(v, "name") and isinstance(v.name, str):
        return v.name
    s = str(v)
    if len(s) > _TRUNCATE_AT:
        s = s[:_TRUNCATE_AT] + "...<truncated>"
    return s


def _cleaned(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean params for storage, expanding top-level ``dims()`` objects.

    A top-level value defining ``dims()`` (see :func:`dims_of`) is replaced
    by its dims entries, so dimensions stay flat scalar columns (e.g.
    ``dataset``/``reaction``/``reaction_file``) instead of one verbose
    stringified blob. The raw object is still available to the
    :class:`~tracking.resolve.Resolver` -- this only affects what is
    tracked. Explicit keys in ``params`` win on collision, processed in
    order.
    """
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (dict, list, tuple)):
            continue
        d = dims_of(v)
        if d is not None:
            for dk, dv in d.items():
                out[str(dk)] = to_bsonable(dv)
    # Explicit scalar keys win over expanded dims on collision: re-apply
    # any plain (non-dims) entries last.
    for k, v in params.items():
        if isinstance(v, (dict, list, tuple)) or dims_of(v) is None:
            out[str(k)] = to_bsonable(v)
    return out


def spec_hash(params: Dict[str, Any]) -> str:
    """Stable SHA-256 hex digest over the cleaned dimensions in ``params``.

    The hash covers exactly what dedup compares: :func:`_cleaned` output
    minus ``benchmark_name`` (same spec under another name counts as the
    same run) and minus ``*_factory`` markers (added by the driver after
    the dedup check, so absent at query time). Canonical JSON
    (``sort_keys``, compact separators) keeps it independent of dict
    ordering; values are already BSON/JSON-safe from :func:`_cleaned`,
    with ``default=str`` as a backstop.
    """
    cleaned = _cleaned(params)
    payload = {
        k: v for k, v in cleaned.items()
        if k != "benchmark_name" and not k.endswith("_factory")
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Backend lookups for the dedup key
# ---------------------------------------------------------------------------

_HASH_INDEX_ENSURED: set[str] = set()


def _ensure_hash_index(coll, namespace: str) -> None:
    """Best-effort compound index for the dedup lookup (once per process)."""
    if namespace in _HASH_INDEX_ENSURED:
        return
    try:
        coll.create_index([("config.spec_hash", 1), ("status", 1)])
        _HASH_INDEX_ENSURED.add(namespace)
    except Exception:
        logger.debug("could not create spec_hash index on %s", namespace, exc_info=True)


# Incremental scan state for the file backend: path -> [mtime, size, offset, hashes].
_FILE_HASH_CACHE: dict[str, list] = {}


def _successful_file_hashes(file_path) -> set[str]:
    """Hashes of SUCCESS rows in the JSONL log, reading only new bytes."""
    path = Path(file_path)
    try:
        stat = path.stat()
    except OSError:
        return set()
    entry = _FILE_HASH_CACHE.get(str(path))
    if entry is not None and entry[0] == stat.st_mtime and entry[1] == stat.st_size:
        return entry[3]
    if entry is not None and stat.st_size >= entry[2]:
        offset, hashes = entry[2], entry[3]
    else:  # new, rotated, or truncated file: scan from the start
        offset, hashes = 0, set()
    try:
        with open(path, encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a half-written last line from a live run
                if (row.get("metrics") or {}).get("status") == "SUCCESS" and row.get("spec_hash"):
                    hashes.add(row["spec_hash"])
            offset = f.tell()
    except OSError:
        return hashes
    _FILE_HASH_CACHE[str(path)] = [stat.st_mtime, stat.st_size, offset, hashes]
    return hashes
