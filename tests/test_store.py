from tracking.store import LocalStore, write_tree_archive
from tracking.store_s3 import S3Store


# ---------------------------------------------------------------------------
# LocalStore
# ---------------------------------------------------------------------------

def test_local_store_get_file_materializes_a_copy(tmp_path):
    store = LocalStore(tmp_path / "store")
    sha = store.put_bytes(b"hello")

    assert store.get_file(sha) == store.object_path(sha)

    dest = tmp_path / "out" / "copy.bin"
    returned = store.get_file(sha, dest)
    assert returned == dest
    assert dest.read_bytes() == b"hello"


def test_local_store_uri_is_a_file_uri(tmp_path):
    store = LocalStore(tmp_path / "store")
    sha = store.put_bytes(b"hello")
    uri = store.uri(sha)
    assert uri.startswith("file://")
    assert uri.endswith(f"/{sha[:2]}/{sha}")


# ---------------------------------------------------------------------------
# S3Store (fake boto3 client, no network/AWS/MinIO dependency)
# ---------------------------------------------------------------------------

class _NotFound(Exception):
    def __init__(self):
        self.response = {"Error": {"Code": "404"}}


class _FakeS3Client:
    """In-memory stand-in for the subset of boto3's S3 client we use."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.put_calls = 0

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NotFound()
        return {}

    def put_object(self, Bucket, Key, Body):
        self.put_calls += 1
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.objects[Key])}


def test_s3_store_roundtrip_and_dedup(tmp_path, monkeypatch):
    # botocore's real ClientError requires a response/operation_name; the
    # fake's _NotFound only needs to look like one to S3Store's except clause.
    import tracking.store_s3 as store_s3

    monkeypatch.setattr(store_s3, "ClientError", _NotFound)

    client = _FakeS3Client()
    store = S3Store("bucket", prefix="proj", client=client, cache_dir=tmp_path / "cache")

    sha = store.put_bytes(b"hello")
    assert store.has(sha)
    assert store.get_bytes(sha) == b"hello"
    assert client.put_calls == 1

    # Re-putting identical content does not re-upload.
    store.put_bytes(b"hello")
    assert client.put_calls == 1

    assert store.uri(sha) == f"s3://bucket/proj/objects/{sha[:2]}/{sha}"

    dest = tmp_path / "materialized.bin"
    assert store.get_file(sha, dest) == dest
    assert dest.read_bytes() == b"hello"


def test_s3_store_put_tree_matches_local_archive(tmp_path, monkeypatch):
    import tracking.store_s3 as store_s3

    monkeypatch.setattr(store_s3, "ClientError", _NotFound)

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("a", encoding="utf-8")
    tree = {"tree_hash": "deadbeef", "files": ["a.txt"]}

    client = _FakeS3Client()
    store = S3Store("bucket", client=client, cache_dir=tmp_path / "cache")
    result = store.put_tree(source, tree)

    assert result["path"] == "trees/deadbeef.tar.gz"
    assert store.has_tree("deadbeef")

    local_archive = tmp_path / "local.tar.gz"
    write_tree_archive(source, tree, (), local_archive)
    from tracking.store import hash_file

    assert result["sha256"] == hash_file(local_archive)
