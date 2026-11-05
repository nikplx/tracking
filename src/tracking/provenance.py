"""Provenance capture for a single run.

Pure helpers with no MongoDB dependency: turning the current git/working-tree
state, the dependency modules a run actually uses, the Python environment, and
the input files into a small, hashable record that travels with the run
document.

The record answers two questions separately:

* *should this run be recomputed?* — see :func:`make_run_key`, which combines
  the parameter spec with the code and input fingerprints; and
* *what produced this number?* — the full record, archived at freeze time.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

from tracking.store import DEFAULT_EXCLUDES, Store, hash_bytes, hash_file

# Packages that can change numerical results and are therefore worth recording.
DEFAULT_PACKAGES = ("tracking", "pythc", "numpy", "scipy", "pandas", "pymongo")


def tracking_version() -> str:
    try:
        return metadata.version("tracking")
    except metadata.PackageNotFoundError:  # editable/source checkout
        return "0.1.0"


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------

def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    return hash_bytes(_canonical(payload))


def deps_digest(deps: Iterable[dict]) -> str:
    return _digest([(d["name"], d["sha256"]) for d in deps])


def inputs_digest(inputs: Iterable[dict]) -> str:
    return _digest([(i["name"], i["sha256"]) for i in inputs])


def make_run_key(spec: str, deps_hash: str, inputs_hash: str) -> str:
    """Stable identity of *this computation*.

    A re-run of the same spec under the same code and inputs replaces its own
    document; a re-run under changed code or inputs lands alongside it. The
    environment is deliberately excluded — see ``Project.status``.
    """
    return _digest([spec, deps_hash, inputs_hash])


# ---------------------------------------------------------------------------
# Dependency hashing
# ---------------------------------------------------------------------------

def _impl_name(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}:{getattr(obj, '__qualname__', repr(obj))}"


def hash_callable(fn: Callable[..., Any]) -> str:
    """Hash a function/class by its source, falling back to its module file."""
    try:
        source = inspect.getsource(inspect.unwrap(fn))
    except (OSError, TypeError):
        source = None
    if source is not None:
        return hash_bytes(source.encode("utf-8"))
    path = inspect.getsourcefile(fn)
    if path:
        return hash_file(path)
    return hash_bytes(_impl_name(fn).encode("utf-8"))


def _import_attr(module_name: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_dep(spec: str | Path) -> dict:
    """Turn a dependency spec into ``{"name", "sha256"}``.

    Accepted forms:

    * ``"pkg.module"`` — module source file (the default seed granularity);
    * ``"pkg.module:qualname"`` — function/class source, module file as fallback;
    * ``"path/to/file.py"`` — that file's bytes.
    """
    if isinstance(spec, Path):
        return {"name": str(spec), "sha256": hash_file(spec)}

    text = str(spec)
    if text.endswith(".py") and os.path.exists(text):
        return {"name": text, "sha256": hash_file(text)}

    module_name, _, qualname = text.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # A factory defined in a script/test module that is not importable
            # by name: record a stable marker rather than failing the run.
            return {"name": text, "sha256": hash_bytes(text.encode("utf-8"))}

    if qualname:
        try:
            return {
                "name": text,
                "sha256": hash_callable(_import_attr(module_name, qualname)),
            }
        except (AttributeError, ImportError):
            pass

    path = getattr(module, "__file__", None)
    if path is None:  # namespace or builtin package
        return {"name": text, "sha256": hash_bytes(text.encode("utf-8"))}
    return {"name": text, "sha256": hash_file(path)}


def seed_deps(factories: dict[str, Any]) -> list[str]:
    """Module-level dependency specs for every registered factory."""
    modules = {
        getattr(f, "__module__", None) for f in factories.values()
    }
    return sorted(m for m in modules if m)


# ---------------------------------------------------------------------------
# Source tree / git
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _walk_files(root: Path, excludes: Iterable[str]) -> list[str]:
    excluded = set(DEFAULT_EXCLUDES) | set(excludes)
    leaves: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded)
        for name in filenames:
            rel = Path(dirpath, name).relative_to(root).as_posix()
            leaves.append(rel)
    return sorted(leaves)


def tree_fingerprint(
    root: Path | str = ".",
    excludes: Iterable[str] = (),
) -> dict:
    """Describe the current source tree.

    Returns ``{commit, branch, dirty, tag, tree_hash, files}``. With git
    available, unchanged tracked files contribute their index object id (no
    file read) and only modified/untracked files are hashed by content, so
    this stays cheap even in a large repository. Without git, a plain walk
    excluding the usual cache/venv directories is used and the git fields are
    ``None``.
    """
    root = Path(root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    tag = _git(root, "describe", "--tags", "--exact-match")
    status = _git(root, "status", "--porcelain", "-z")
    tracked = _git(root, "ls-files", "-s", "-z")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")

    excluded = set(excludes)
    if tracked is None:
        files = _walk_files(root, excludes)
        entries = [
            (rel, hash_file(root / rel)) for rel in files if (root / rel).is_file()
        ]
        dirty = None
    else:
        changed = _parse_status_paths(status or "")
        entries: list[tuple[str, str]] = []
        tracked_files: list[str] = []
        for record in (tracked or "").split("\0"):
            if not record:
                continue
            meta, _, rel = record.partition("\t")
            if not rel or _matches_exclude(rel, excluded):
                continue
            fields = meta.split()
            oid = fields[1] if len(fields) > 1 else ""
            path = root / rel
            if rel in changed:
                entries.append((rel, hash_file(path) if path.is_file() else "deleted"))
            else:
                entries.append((rel, oid))
            if path.is_file():
                tracked_files.append(rel)

        for rel in (untracked or "").split("\0"):
            if not rel or _matches_exclude(rel, excluded):
                continue
            path = root / rel
            if path.is_file():
                entries.append((rel, hash_file(path)))
                tracked_files.append(rel)

        entries.sort()
        files = sorted(set(tracked_files))
        dirty = bool((status or "").strip("\0"))

    entries.sort()
    return {
        "commit": (commit or "").strip() or None,
        "branch": (branch or "").strip() or None,
        "tag": (tag or "").strip() or None,
        "dirty": dirty,
        "tree_hash": _digest(entries),
        "files": files,
    }


def _parse_status_paths(out: str) -> set[str]:
    """Paths mentioned by ``git status --porcelain -z`` (new side of renames)."""
    tokens = [t for t in out.split("\0") if t]
    paths: set[str] = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if len(token) > 3:
            paths.add(token[3:])
            if token[0] in "RC" or token[1] in "RC":
                i += 2  # the following token is the rename/copy source
                continue
        i += 1
    return paths


def _matches_exclude(rel: str, excludes: Iterable[str]) -> bool:
    parts = Path(rel).parts
    for excluded in excludes:
        root_parts = Path(excluded).parts
        if parts[: len(root_parts)] == root_parts:
            return True
    return False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def env_fingerprint(packages: Iterable[str] | None = None) -> dict:
    names = tuple(packages) if packages is not None else DEFAULT_PACKAGES
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue  # optional package for this workflow
    payload = {"python": platform.python_version(), "packages": versions}
    return {**payload, "hash": _digest(payload)}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def collect_inputs(
    paths: Iterable[str | Path],
    store: Store,
) -> list[dict]:
    """Hash and copy each input into the content-addressed store."""
    records: dict[tuple[str, str], dict] = {}
    for item in paths:
        path = Path(item)
        sha = hash_file(path)
        store.put_file(path)
        stored = store.get_file(sha)
        record = {
            "name": path.name,
            "path": str(path),
            "sha256": sha,
            "object": f"objects/{sha[:2]}/{stored.name}",
        }
        records[(record["name"], record["path"])] = record
    return [records[key] for key in sorted(records)]
