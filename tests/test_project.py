import pytest

import tracking.experiment as experiment
import tracking.project as project_module
from tracking import Experiment, observe
from tracking.project import FrozenProjectError, Project
from tracking.provenance import tree_fingerprint
from tracking.store import LocalStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeCollection:
    """In-memory collection keyed by the (spec_hash, run_key) write filter."""

    def __init__(self):
        self.docs = {}

    def create_index(self, *args, **kwargs):
        return None

    def find_one(self, query, projection=None):
        for doc in self.docs.values():
            if all(doc.get(key) == value for key, value in query.items()):
                return {"_id": "fake"}
        return None

    def find(self, query=None, projection=None):
        query = query or {}
        return [
            doc
            for doc in self.docs.values()
            if all(doc.get(key) == value for key, value in query.items())
        ]

    def replace_one(self, filt, doc, upsert=False):
        self.docs[(filt.get("spec_hash"), filt.get("run_key"))] = doc

    def update_many(self, query, update):
        matched = 0
        for doc in self.find(query):
            doc.update(update.get("$set", {}))
            matched += 1
        return matched


class _FakeDB:
    def __init__(self):
        self.requested = []

    def __getitem__(self, name):
        self.requested.append(name)
        return name


class _FakeClient:
    def __init__(self):
        self.dbs = {}

    def __getitem__(self, name):
        return self.dbs.setdefault(name, _FakeDB())


# ---------------------------------------------------------------------------
# Project / connection plumbing
# ---------------------------------------------------------------------------

def test_get_runs_collection_honours_db(tmp_path, monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(experiment, "get_mongo_client", lambda uri=None: client)
    monkeypatch.delenv("TRACKING_PROJECT", raising=False)
    project_module.reset_default_project()

    assert experiment.get_runs_collection("runs", db="other") == "runs"
    assert "other" in client.dbs

    assert experiment.get_runs_collection("runs") == "runs"
    assert project_module.default_project().db == "tracking"


def test_tracking_project_env_selects_default_db(monkeypatch):
    monkeypatch.setenv("TRACKING_PROJECT", "paper_2026")
    project_module.reset_default_project()
    try:
        assert project_module.default_project().db == "paper_2026"
    finally:
        monkeypatch.delenv("TRACKING_PROJECT")
        project_module.reset_default_project()


def test_project_store_uses_configured_root(tmp_path):
    project = Project("p", store_root=tmp_path / "store")
    assert project.store.root == tmp_path / "store"
    assert isinstance(project.store, LocalStore)


# ---------------------------------------------------------------------------
# Experiment provenance capture
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_collection(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(
        experiment, "get_runs_collection", lambda name, **kwargs: collection
    )
    return collection


def _run(collection, tmp_path, deps, name="run", body=None):
    run = Experiment(
        name,
        {"x": 1},
        project=Project("p", store_root=tmp_path / "store"),
        deps=deps,
    )
    run._provenance_root = tmp_path
    with run:
        if body is not None:
            body(run)
        else:
            observe.log_metric("energy", 1.5)
    return run


def test_compact_run_records_provenance(fake_collection, tmp_path):
    _run(fake_collection, tmp_path, ["tracking.provenance"])

    doc = next(iter(fake_collection.docs.values()))
    provenance = doc["provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["code"]["deps"][0]["name"] == "tracking.provenance"
    assert provenance["code"]["deps_hash"]
    assert provenance["env"]["hash"]
    assert provenance["inputs"] == []
    assert provenance["inputs_hash"]
    assert doc["run_key"]
    assert doc["tier"] == "draft"
    assert doc["meta"]["schema_version"] == 2


def test_successful_run_snapshots_source_tree(fake_collection, tmp_path):
    run = _run(fake_collection, tmp_path, [])
    doc = next(iter(fake_collection.docs.values()))
    snapshot = doc["provenance"]["code"]["snapshot"]
    assert snapshot is not None
    assert snapshot["path"] == f"trees/{doc['provenance']['code']['tree_hash']}.tar.gz"
    assert (tmp_path / "store" / snapshot["path"]).is_file()
    assert run.project.store.has_tree(doc["provenance"]["code"]["tree_hash"])


def test_failed_run_has_no_snapshot(fake_collection, tmp_path):
    run = Experiment(
        "run", {"x": 1}, project=Project("p", store_root=tmp_path / "store")
    )
    run._provenance_root = tmp_path
    with pytest.raises(RuntimeError):
        with run:
            raise RuntimeError("boom")

    doc = next(iter(fake_collection.docs.values()))
    assert doc["status"] == "FAILED"
    assert doc["provenance"]["code"]["snapshot"] is None


def test_same_computation_replaces_its_own_doc(fake_collection, tmp_path):
    _run(fake_collection, tmp_path, ["tracking.provenance"])
    _run(fake_collection, tmp_path, ["tracking.provenance"])
    assert len(fake_collection.docs) == 1


def test_changed_dependency_appends_instead_of_overwriting(fake_collection, tmp_path):
    _run(fake_collection, tmp_path, ["tracking.provenance"])
    _run(fake_collection, tmp_path, ["tracking.store"])

    assert len(fake_collection.docs) == 2
    keys = {doc["run_key"] for doc in fake_collection.docs.values()}
    assert len(keys) == 2


def test_graph_is_persisted(fake_collection, tmp_path):
    graph = {"run": {"impl": "benchmarks.runners:THCRun", "deps": ["mol"]}}

    def body(run):
        run.set_graph(graph)

    _run(fake_collection, tmp_path, [], body=body)
    doc = next(iter(fake_collection.docs.values()))
    assert doc["provenance"]["code"]["graph"] == graph


def test_explicit_project_rejects_writes_when_frozen(fake_collection, tmp_path):
    project = Project("p", store_root=tmp_path / "store")
    project._manifest = {"_kind": "tracking.manifest", "frozen": True}
    project._manifest_at = 1e18  # never expire in this test

    run = Experiment("run", {"x": 1}, project=project)
    run._provenance_root = tmp_path
    with pytest.raises(FrozenProjectError):
        with run:
            pass


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_store_objects_are_idempotent(tmp_path):
    store = LocalStore(tmp_path / "store")
    sha = store.put_bytes(b"hello")
    again = store.put_bytes(b"hello")
    assert sha == again
    assert store.get_bytes(sha) == b"hello"


def test_store_tree_is_deterministic(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("a", encoding="utf-8")
    (source / "sub").mkdir()
    (source / "sub" / "b.txt").write_text("b", encoding="utf-8")

    tree = tree_fingerprint(source)
    store = LocalStore(tmp_path / "store")

    first = store.put_tree(source, tree)
    archive = store.tree_path(tree["tree_hash"])
    before = archive.read_bytes()

    archive.unlink()
    second = store.put_tree(source, tree)

    assert second == first
    assert archive.read_bytes() == before
