"""S3-compatible object storage backend (optional; requires ``boto3``).

Implements the same protocol as :class:`tracking.store.LocalStore`
(see :class:`tracking.store.Store`), so :class:`~tracking.project.Project`
can point at a shared POSIX directory or an S3-compatible bucket (AWS,
MinIO, Ceph, ...) without any other code noticing which one is in use.

Install with the ``tracking[s3]`` extra. This module is only imported when
``TRACKING_STORE``/``Project.store_root`` actually names an ``s3://`` URI,
so ``boto3`` stays optional for everyone else.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from tracking.store import hash_bytes, hash_file, normalize_suffix, write_tree_archive

_RETRY = dict(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))


class S3Store:
    """Content-addressed objects under ``s3://<bucket>/<prefix>``.

    A local read-through cache (``cache_dir``, defaulting to a directory
    under the system temp dir) backs every read so repeated access to the
    same object doesn't re-download it; uploads are still deduplicated
    against the bucket itself via ``has``/``has_tree``.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client=None,
        cache_dir: Path | str | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.gettempdir()) / "tracking_s3_cache"
        )
        # sha -> key, so a suffixed object written in this process can be
        # found again without a bucket listing. Real buckets additionally
        # support discovery via ``list_objects_v2`` below.
        self._known: dict[str, str] = {}

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    # -- keys --------------------------------------------------------------
    def _key(self, *parts: str) -> str:
        return "/".join(p for p in (self.prefix, *parts) if p)

    def _object_key(self, sha: str, suffix: str = "") -> str:
        return self._key("objects", sha[:2], f"{sha}{normalize_suffix(suffix)}")

    def _find_object_key(self, sha: str, suffix: str = "") -> str:
        """Locate ``sha`` in the bucket, discovering its stored extension."""
        if suffix:
            return self._object_key(sha, suffix)
        known = self._known.get(sha)
        if known is not None:
            return known
        base = self._object_key(sha)
        if self._head(base):
            self._known[sha] = base
            return base
        lister = getattr(self.client, "list_objects_v2", None)
        if lister is not None:
            try:
                prefix = self._key("objects", sha[:2], sha)
                response = lister(Bucket=self.bucket, Prefix=prefix)
                keys = sorted(
                    item["Key"]
                    for item in response.get("Contents", ())
                    if item["Key"] == prefix or item["Key"].startswith(f"{prefix}.")
                )
                if keys:
                    self._known[sha] = keys[0]
                    return keys[0]
            except Exception:
                pass
        return base

    def _tree_key(self, tree_hash: str) -> str:
        return self._key("trees", f"{tree_hash}.tar.gz")

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key

    def _write_cache(self, key: str, data: bytes) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # -- raw S3 calls, retried against flaky networks -----------------------
    @retry(**_RETRY)
    def _head(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey"):
                return False
            raise

    @retry(**_RETRY)
    def _put_object(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    @retry(**_RETRY)
    def _get_object(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    # -- objects -------------------------------------------------------------
    def has(self, sha: str) -> bool:
        return self._head(self._find_object_key(sha))

    def put_bytes(self, data: bytes, *, suffix: str = "") -> str:
        sha = hash_bytes(data)
        existing = self._find_object_key(sha)
        if self._head(existing):
            self._known[sha] = existing
            key = existing
        else:
            key = self._object_key(sha, suffix)
            self._put_object(key, data)
            self._known[sha] = key
        self._write_cache(key, data)
        return sha

    def put_file(self, path: Path | str, *, suffix: str | None = None) -> str:
        path = Path(path)
        if suffix is None:
            suffix = path.suffix
        return self.put_bytes(path.read_bytes(), suffix=suffix)

    def get_bytes(self, sha: str) -> bytes:
        key = self._find_object_key(sha)
        cached = self._cache_path(key)
        if cached.is_file():
            return cached.read_bytes()
        data = self._get_object(key)
        self._write_cache(key, data)
        return data

    def get_file(
        self, sha: str, dest: Path | str | None = None, *, suffix: str = ""
    ) -> Path:
        key = self._find_object_key(sha, suffix)
        cached = self._cache_path(key)
        if not cached.is_file():
            data = self._get_object(key)
            self._write_cache(key, data)
        if dest is None:
            return cached
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cached.read_bytes())
        return dest

    def uri(self, sha: str, *, suffix: str = "") -> str:
        return f"s3://{self.bucket}/{self._find_object_key(sha, suffix)}"

    def object_key(self, sha: str, *, suffix: str = "") -> str:
        """``objects/<2>/<sha><ext>``, relative to this store's prefix."""
        key = self._find_object_key(sha, suffix)
        prefix = f"{self.prefix}/" if self.prefix else ""
        return key[len(prefix) :] if prefix and key.startswith(prefix) else key

    # -- source trees ----------------------------------------------------
    def has_tree(self, tree_hash: str) -> bool:
        return self._head(self._tree_key(tree_hash))

    def put_tree(
        self,
        root: Path | str,
        tree: dict,
        excludes: Iterable[str] = (),
    ) -> dict:
        tree_hash = tree["tree_hash"]
        key = self._tree_key(tree_hash)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".tar.gz")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            write_tree_archive(root, tree, excludes, tmp_path)
            sha256 = hash_file(tmp_path)
            if not self._head(key):
                self._put_object(key, tmp_path.read_bytes())
        finally:
            tmp_path.unlink(missing_ok=True)
        return {"path": f"trees/{tree_hash}.tar.gz", "sha256": sha256}
