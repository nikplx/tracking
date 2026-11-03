"""Projects: one MongoDB database per paper/investigation.

A :class:`Project` fixes three things that used to be ambient process state:

* which database the runs live in (collections are sweeps within it),
* which content-addressed store backs provenance, and
* whether the project is frozen and must reject new writes.

It stays deliberately small: no registry, no plugin hooks. ``default_project``
resolves the backward-compatible default from the environment.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracking.provenance import env_fingerprint, resolve_dep
from tracking.store import hash_file

logger = logging.getLogger(__name__)

_MANIFEST_TTL = 30.0  # seconds; a frozen project need not hit Mongo on every checkpoint


class FrozenProjectError(RuntimeError):
    """Raised when a run tries to write into a frozen project."""


def default_db_name() -> str:
    return os.environ.get("TRACKING_MONGO_DB", "tracking")


def default_store_root() -> str | Path:
    """Store location, honouring ``TRACKING_STORE`` then ``DATA_DIR``.

    ``TRACKING_STORE`` may be a plain path or a ``s3://bucket/prefix`` URI
    (see :func:`make_store`); URIs are returned as-is, never wrapped in
    ``Path``, which would otherwise mangle the ``://`` separator.
    ``DATA_DIR`` is read from :mod:`tracking.experiment` when available so
    tests and callers that monkeypatch it keep working.
    """
    env = os.environ.get("TRACKING_STORE")
    if env:
        return env if "://" in env else Path(env)
    try:
        from tracking.experiment import DATA_DIR

        return Path(DATA_DIR) / "tracking_store"
    except Exception:
        return Path(os.environ.get("DATA_DIR", ".")) / "tracking_store"


def make_store(store_root: str | Path):
    """Build the :class:`~tracking.store.Store` backend named by ``store_root``.

    An ``s3://bucket/prefix`` string dispatches to
    :class:`tracking.store_s3.S3Store` (which requires the ``tracking[s3]``
    extra); anything else is a local directory, backed by
    :class:`tracking.store.LocalStore`.
    """
    text = str(store_root)
    if text.startswith("s3://"):
        from tracking.store_s3 import S3Store

        bucket, _, prefix = text[len("s3://") :].partition("/")
        return S3Store(bucket, prefix)
    from tracking.store import LocalStore

    return LocalStore(store_root)


@dataclass
class Project:
    """One paper / investigation, backed by one MongoDB database."""

    name: str
    db: str | None = None
    uri: str | None = None
    store_root: Path | str | None = None
    writable: bool = False
    tree_excludes: tuple[str, ...] = ()

    _store: Any = field(default=None, init=False, repr=False)
    _manifest: dict | None = field(default=None, init=False, repr=False)
    _manifest_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.db is None:
            self.db = self.name
        if self.store_root is None:
            self.store_root = default_store_root()

    # -- connection ------------------------------------------------------
    @property
    def store(self):
        if self._store is None:
            self._store = make_store(self.store_root)
        return self._store

    def artifact_uri(self, sha: str) -> str:
        return self.store.uri(sha)

    def client(self):
        from tracking.experiment import get_mongo_client

        return get_mongo_client(self.uri)

    def collection(self, name: str):
        return self.client()[self.db][name or "runs"]

    # -- reads -----------------------------------------------------------
    def docs(self, collection: str, query: dict | None = None) -> list[dict]:
        return list(self.collection(collection).find(query or {}))

    def load(self, collection: str, query: dict | None = None):
        from tracking.load import load_mongo_data

        return load_mongo_data(self.collection(collection), query or {})

    def collection_names(self) -> list[str]:
        """Non-internal collection names in this project's database."""
        names = self.client()[self.db].list_collection_names()
        return sorted(
            n for n in names if not n.startswith("_") and not n.startswith("system.")
        )

    def load_artifact(
        self, descriptor: dict | str, dest: Path | str | None = None
    ) -> bytes | Path:
        """Fetch an artifact previously written by ``Experiment.add_artifact``.

        ``descriptor`` is either the dict returned by ``add_artifact``/found
        in a run document's ``artifacts`` list, or a raw sha256. Returns raw
        bytes, or a materialized file path when ``dest`` is given.
        """
        sha = descriptor["sha256"] if isinstance(descriptor, dict) else descriptor
        if dest is None:
            return self.store.get_bytes(sha)
        return self.store.get_file(sha, dest)

    def artifacts(self, collection: str | None = None, query: dict | None = None):
        """One row per artifact across matching runs, newest fields flat.

        Mirrors :meth:`status`'s shape convention: iterates the same
        ``(collection, doc)`` pairs and explodes each doc's ``artifacts``
        list, tagging every row with the run it came from.
        """
        import pandas as pd

        rows: list[dict] = []
        for name, doc in self._iter_docs(collection, query):
            for artifact in doc.get("artifacts") or []:
                rows.append(
                    {
                        **artifact,
                        "collection": name,
                        "spec_hash": doc.get("spec_hash"),
                        "run_key": doc.get("run_key"),
                    }
                )
        return pd.DataFrame(rows)

    def status(self, collection: str | None = None, query: dict | None = None):
        """Runs plus a computed ``validity`` column.

        Validity is a *report*, never a mutation: ``CURRENT``,
        ``STALE_CODE``, ``STALE_ENV``, ``STALE_INPUT``, ``INVALIDATED`` or
        ``UNKNOWN`` (pre-provenance documents). Use :meth:`invalidate` to
        mark a subset deliberately and ``rerun="stale"`` to act on it.
        """
        import pandas as pd

        from tracking.experiment import normalize_doc

        current_env = env_fingerprint()
        dep_cache: dict[str, str | None] = {}
        rows: list[dict] = []
        for name, doc in self._iter_docs(collection, query):
            row = normalize_doc(doc)
            row["collection"] = name
            validity, detail = self._classify(doc, current_env, dep_cache)
            row["validity"] = validity
            row["validity_detail"] = detail
            rows.append(row)
        return pd.DataFrame(rows)

    def invalidate(self, collection: str, query: dict | None, reason: str) -> int:
        """Mark matching runs stale. Never deletes or recomputes."""
        result = self.collection(collection).update_many(
            query or {},
            {
                "$set": {
                    "invalidated": {
                        "at": datetime.now(timezone.utc),
                        "reason": reason,
                    }
                }
            },
        )
        return result.modified_count

    def _iter_docs(self, collection: str | None, query: dict | None):
        if collection is not None:
            for doc in self.docs(collection, query):
                yield collection, doc
            return
        for name in self.collection_names():
            for doc in self.docs(name, query):
                yield name, doc

    def _classify(
        self,
        doc: dict,
        current_env: dict,
        dep_cache: dict[str, str | None],
    ) -> tuple[str, str]:
        if doc.get("invalidated"):
            reason = (doc.get("invalidated") or {}).get("reason") or "invalidated"
            return "INVALIDATED", reason

        provenance = doc.get("provenance")
        if not provenance:
            return "UNKNOWN", "no provenance recorded"

        code = provenance.get("code") or {}
        for dep in code.get("deps") or []:
            name = dep.get("name")
            if not name:
                continue
            if name not in dep_cache:
                try:
                    dep_cache[name] = resolve_dep(name)["sha256"]
                except Exception:
                    logger.debug("could not re-hash dependency %s", name, exc_info=True)
                    dep_cache[name] = None
            current = dep_cache[name]
            if current is not None and current != dep.get("sha256"):
                return "STALE_CODE", f"{name} changed"

        stored_env = provenance.get("env") or {}
        if stored_env.get("hash") and stored_env["hash"] != current_env["hash"]:
            return "STALE_ENV", "environment changed"

        for record in provenance.get("inputs") or []:
            sha = record.get("sha256")
            if not sha:
                continue
            if not self.store.has(sha):
                return "STALE_INPUT", f"{record.get('name', sha)} object missing"
            path = record.get("path")
            if path and Path(path).is_file() and hash_file(path) != sha:
                return "STALE_INPUT", f"{record.get('name', sha)} changed"

        return "CURRENT", ""

    # -- freeze guard ----------------------------------------------------
    def manifest(self) -> dict | None:
        now = time.monotonic()
        if self._manifest is not None and now - self._manifest_at < _MANIFEST_TTL:
            return self._manifest
        try:
            self._manifest = self.client()[self.db]["_manifest"].find_one(
                {"_kind": "tracking.manifest"}
            )
        except Exception:
            logger.debug("could not read project manifest", exc_info=True)
            return self._manifest
        self._manifest_at = now
        return self._manifest

    def is_frozen(self) -> bool:
        manifest = self.manifest()
        return bool(manifest and manifest.get("frozen"))

    def assert_writable(self) -> None:
        if self.writable:
            return
        if self.is_frozen():
            raise FrozenProjectError(
                f"project {self.name!r} is frozen (db {self.db!r}); "
                "construct Project(..., writable=True) for a deliberate revision"
            )


_DEFAULT_PROJECT: Project | None = None


def default_project() -> Project:
    """The project used when the caller does not name one.

    ``TRACKING_PROJECT`` wins if set, otherwise the historical
    ``TRACKING_MONGO_DB`` database name is used so existing deployments and
    notebooks keep writing where they always did.
    """
    global _DEFAULT_PROJECT
    if _DEFAULT_PROJECT is None:
        name = os.environ.get("TRACKING_PROJECT")
        if name:
            _DEFAULT_PROJECT = Project(name)
        else:
            _DEFAULT_PROJECT = Project(default_db_name(), db=default_db_name())
    return _DEFAULT_PROJECT


def reset_default_project() -> None:
    """Drop the cached default (for tests that change the environment)."""
    global _DEFAULT_PROJECT
    _DEFAULT_PROJECT = None
