import json
from pathlib import Path

import pandas as pd

from tracking.experiment import normalize_doc


def load_mongo_data(collection, query):
    """Fetch a Mongo collection and return a normalized ``pandas.DataFrame``."""
    docs = list(collection.find(query))
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame([normalize_doc(d) for d in docs])


def _read_file_docs(source) -> list[dict]:
    """Read run docs from a ``FileCollection``, a ``.json`` file/dir, or JSONL."""
    from tracking.file_backend import FileCollection

    if isinstance(source, FileCollection):
        return source.find({})

    path = Path(source)
    docs: list[dict] = []
    if path.is_dir():
        items = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
        if not items:
            # A file-backend root (as opposed to one collection) contains a
            # subdirectory per collection; read them all.
            for child in sorted(p for p in path.iterdir() if p.is_dir()):
                docs.extend(_read_file_docs(child))
            return docs
        for item in items:
            if item.suffix == ".json":
                try:
                    value = json.loads(item.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                docs.extend(value if isinstance(value, list) else [value])
            else:
                docs.extend(_read_jsonl(item))
        return docs
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    return _read_jsonl(path)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a half-written last line from a live run
    return rows


def load_file_data(
    source, query: dict | None = None, collection: str | None = None
) -> pd.DataFrame:
    """Read the local file backend into a normalized ``pandas.DataFrame``.

    ``source`` is a :class:`tracking.project.Project` (optionally with a
    ``collection`` name), a :class:`tracking.file_backend.FileCollection`, a
    directory of per-run ``.json`` documents, a single ``.json`` document, or
    a JSONL log. The result matches :func:`load_mongo_data` (``dimensions_*``
    / ``metrics_*`` / ``provenance_*`` columns).
    """
    from tracking.file_backend import _matches
    from tracking.project import Project

    if isinstance(source, Project):
        source = source.file_collection(collection)

    docs = [doc for doc in _read_file_docs(source) if _matches(doc, query)]
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame([normalize_doc(doc) for doc in docs])


def load_artifacts(
    project,
    collection: str | None = None,
    query: dict | None = None,
    dest: Path | str | None = None,
):
    """Read a project's artifact descriptors as a DataFrame with local paths.

    Convenience wrapper around :meth:`tracking.project.Project.load_artifacts`
    so artifacts are reachable the same way runs are via
    :func:`load_json_data`/:func:`load_mongo_data`.
    """
    return project.load_artifacts(collection, query=query, dest=dest)


def load_json_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir() or path.suffix == ".json":
        rows = _read_file_docs(path)
    else:
        rows = _read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path} has no complete JSON lines yet.")

    df = pd.json_normalize(rows, sep="_")

    # pygwalker needs scalar cells; stringify anything that came through as
    # a list or dict (e.g. a non-empty `checkpoints` array).
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)

    # Best-effort: parse anything that looks like a timestamp column.
    for col in df.columns:
        if "timestamp" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed

    return df
