"""Content-addressed storage for run provenance.

Everything a run needs to be reproducible later — input files and a tarball of
the source tree — is stored by content hash under a single root, so identical
content is never written twice and every reference in a run document is
verifiable.

``Store`` is the protocol every backend satisfies; :class:`LocalStore` is the
plain-filesystem implementation. ``tracking.store_s3.S3Store`` implements the
same protocol against an S3-compatible bucket.

Local layout::

    <root>/objects/<sha[:2]>/<sha>      # arbitrary blobs (inputs, artifacts)
    <root>/trees/<tree_hash>.tar.gz     # source-tree snapshots, keyed by tree hash

Pure stdlib: :mod:`tarfile` and :mod:`gzip`, no extra dependencies.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

# Directories that are never part of a source snapshot or tree hash.
DEFAULT_EXCLUDES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
    }
)


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------

_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path | str) -> str:
    """SHA-256 of a file's contents, cached by ``(path, mtime_ns, size)``.

    The cache is process-lifetime: sweep processes are short-lived, and the
    same dependency files are re-hashed for every grid point otherwise.
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_HASH_CACHE[key] = digest
    return digest


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Store(Protocol):
    """A content-addressed blob store: local disk, S3, or anything else."""

    def has(self, sha: str) -> bool: ...
    def put_bytes(self, data: bytes) -> str: ...
    def put_file(self, path: Path | str) -> str: ...
    def get_bytes(self, sha: str) -> bytes: ...
    def get_file(self, sha: str, dest: Path | str | None = None) -> Path:
        """Materialize object ``sha`` as a real file and return its path."""
        ...

    def uri(self, sha: str) -> str:
        """A stable locator for object ``sha`` (``file://...``, ``s3://...``)."""
        ...

    def has_tree(self, tree_hash: str) -> bool: ...
    def put_tree(self, root: Path | str, tree: dict, excludes: Iterable[str] = ()) -> dict: ...


# ---------------------------------------------------------------------------
# Tree archives (shared by every backend)
# ---------------------------------------------------------------------------

def _is_excluded(rel: str, excluded_roots: Iterable[str]) -> bool:
    parts = Path(rel).parts
    for root in excluded_roots:
        root_parts = Path(root).parts
        if parts[: len(root_parts)] == root_parts:
            return True
    return False


def write_tree_archive(
    root: Path | str,
    tree: dict,
    excludes: Iterable[str],
    out_path: Path,
) -> None:
    """Write a deterministic ``tar.gz`` of ``tree``'s files to ``out_path``.

    ``tree`` is the result of :func:`tracking.provenance.tree_fingerprint`
    and supplies the member list. Member metadata is zeroed and gzip's mtime
    pinned so the archive is byte-reproducible; shared by every
    :class:`Store` backend's ``put_tree``.
    """
    root = Path(root)
    excluded_roots = tuple(DEFAULT_EXCLUDES) + tuple(excludes)
    leaves = [
        rel
        for rel in tree.get("files", ())
        if not _is_excluded(rel, excluded_roots)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=".tmp-", suffix=".tar.gz")
    os.close(fd)
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for rel in sorted(leaves):
                        path = root / rel
                        if not path.is_file():
                            continue
                        info = tar.gettarinfo(str(path), arcname=rel)
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with open(path, "rb") as f:
                            tar.addfile(info, f)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# LocalStore
# ---------------------------------------------------------------------------

class LocalStore:
    """A directory of immutable, content-addressed objects."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # -- objects ---------------------------------------------------------
    def object_path(self, sha: str) -> Path:
        return self.root / "objects" / sha[:2] / sha

    def has(self, sha: str) -> bool:
        return self.object_path(sha).is_file()

    def put_bytes(self, data: bytes) -> str:
        sha = hash_bytes(data)
        target = self.object_path(sha)
        if target.is_file():
            return sha
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return sha

    def put_file(self, path: Path | str) -> str:
        return self.put_bytes(Path(path).read_bytes())

    def get_bytes(self, sha: str) -> bytes:
        return self.object_path(sha).read_bytes()

    def get_file(self, sha: str, dest: Path | str | None = None) -> Path:
        source = self.object_path(sha)
        if dest is None:
            return source
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        return dest

    def uri(self, sha: str) -> str:
        return self.object_path(sha).resolve().as_uri()

    # -- source trees ----------------------------------------------------
    def tree_path(self, tree_hash: str) -> Path:
        return self.root / "trees" / f"{tree_hash}.tar.gz"

    def has_tree(self, tree_hash: str) -> bool:
        return self.tree_path(tree_hash).is_file()

    def put_tree(
        self,
        root: Path | str,
        tree: dict,
        excludes: Iterable[str] = (),
    ) -> dict:
        """Write a deterministic ``tar.gz`` of the files in ``tree``.

        See :func:`write_tree_archive` for the archive format.
        """
        tree_hash = tree["tree_hash"]
        target = self.tree_path(tree_hash)
        if not target.is_file():
            write_tree_archive(root, tree, excludes, target)
        return {"path": f"trees/{tree_hash}.tar.gz", "sha256": hash_file(target)}
