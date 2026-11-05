"""Local JSON-file backend for run documents.

The file backend is the offline counterpart of the compact Mongo backend: one
self-contained run document per ``.json`` file, grouped by collection::

    <root>/<collection>/<run_key>.json

Files are written atomically (temp file + ``os.replace``), so a checkpoint
from a live run never leaves a half-written document, and the ending is a
proper ``.json`` rather than a directory of Sacred artefacts. The collection
object implements the small subset of the ``pymongo`` API that
:class:`tracking.experiment.Experiment` and :class:`tracking.project.Project`
use, which is what lets ``Project.load``/``status``/``artifacts`` work
unchanged for both backends.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def _resolve(doc: Any, path: str) -> Any:
    current = doc
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches(doc: dict, query: dict | None) -> bool:
    """Mongo-like equality matching, including dotted paths and ``$in``."""
    for key, expected in (query or {}).items():
        actual = _resolve(doc, key)
        if isinstance(expected, dict) and any(
            str(op).startswith("$") for op in expected
        ):
            for op, value in expected.items():
                if op == "$in" and actual not in value:
                    return False
                if op == "$ne" and actual == value:
                    return False
                if op == "$exists" and (actual is not None) != bool(value):
                    return False
                if op not in ("$in", "$ne", "$exists"):
                    raise ValueError(f"unsupported file-backend query operator {op!r}")
        elif actual != expected:
            return False
    return True


class FileCollection:
    """A directory of run documents, exposed like a pymongo collection."""

    def __init__(self, root: Path | str, name: str = "runs"):
        self.root = Path(root)
        self.name = name or "runs"

    @property
    def path(self) -> Path:
        return self.root / self.name

    # -- reading ---------------------------------------------------------
    def _read(self, path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            rows: list[dict] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate a half-written last line
            return rows
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [doc] if isinstance(doc, dict) else []

    def _iter(self) -> Iterator[tuple[Path, dict]]:
        if self.path.is_file():
            for doc in self._read(self.path):
                yield self.path, doc
            return
        if not self.path.is_dir():
            return
        for path in sorted(self.path.glob("*.json")):
            for doc in self._read(path):
                yield path, doc
        for path in sorted(self.path.glob("*.jsonl")):
            for doc in self._read(path):
                yield path, doc

    def find(self, query: dict | None = None, projection=None) -> list[dict]:
        return [doc for _, doc in self._iter() if _matches(doc, query)]

    def find_one(self, query: dict | None = None, projection=None):
        for _, doc in self._iter():
            if _matches(doc, query):
                return doc
        return None

    # -- writing ---------------------------------------------------------
    @staticmethod
    def _key(filt: dict) -> str:
        run_key = filt.get("run_key")
        spec_hash = filt.get("spec_hash")
        if run_key:
            return f"{run_key}.json"
        if spec_hash:
            return f"{spec_hash}.json"
        digest = hashlib.sha256(
            json.dumps(filt, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"{digest}.json"

    def _write(self, path: Path, doc: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, default=str)
                f.write("\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def replace_one(self, filt: dict, doc: dict, upsert: bool = False) -> None:
        target = self.path / self._key(filt)
        if not upsert and not target.is_file():
            return
        self._write(target, doc)

    def update_many(self, query: dict, update: dict) -> Any:
        changes = update.get("$set", {})
        matched = 0

        def assign(doc: dict, dotted: str, value: Any) -> None:
            parts = dotted.split(".")
            current = doc
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value

        for path, doc in self._iter():
            if not _matches(doc, query):
                continue
            for key, value in changes.items():
                assign(doc, key, value)
            if path.suffix == ".json":
                self._write(path, doc)
            matched += 1
        return type("UpdateResult", (), {"modified_count": matched})()

    def create_index(self, *args, **kwargs) -> None:
        return None
