from pathlib import Path

import pytest

import tracking.experiment as experiment
import tracking.provenance as provenance
from tracking import Experiment, Runnable, observe
from tracking.driver import Sweep, execute, run_sweep
from tracking.project import FrozenProjectError, Project


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _resolve(doc, path):
    current = doc
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


class _FakeCollection:
    def __init__(self):
        self.docs = []

    def create_index(self, *args, **kwargs):
        return None

    @staticmethod
    def _matches(doc, query):
        return all(_resolve(doc, key) == value for key, value in query.items())

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if self._matches(doc, query):
                return {"_id": "fake"}
        return None

    def find(self, query=None, projection=None):
        return [doc for doc in self.docs if self._matches(doc, query or {})]

    def replace_one(self, filt, doc, upsert=False):
        for index, existing in enumerate(self.docs):
            if all(existing.get(key) == value for key, value in filt.items()):
                self.docs[index] = doc
                return
        self.docs.append(doc)

    def update_many(self, query, update):
        matched = 0
        for doc in self.find(query):
            doc.update(update.get("$set", {}))
            matched += 1
        return type("Result", (), {"modified_count": matched})()


class _FakeDB:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, _FakeCollection())

    def list_collection_names(self):
        return list(self.collections)


class _FakeClient:
    def __init__(self):
        self.databases = {}

    def __getitem__(self, name):
        return self.databases.setdefault(name, _FakeDB())


@pytest.fixture
def mongo(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(experiment, "get_mongo_client", lambda uri=None: client)
    return client


@pytest.fixture
def fixed_tree(monkeypatch):
    """Keep provenance capture off the real repository for fast unit tests."""
    monkeypatch.setattr(
        experiment,
        "tree_fingerprint",
        lambda root, excludes=(): {
            "commit": "deadbeef",
            "branch": "main",
            "tag": None,
            "dirty": False,
            "tree_hash": "treehash",
            "files": [],
        },
    )


class _Kernel(Runnable):
    def __init__(self, value):
        self.value = value

    def kernel(self) -> None:
        observe.log_metric("value", self.value)


def _project(tmp_path: Path) -> Project:
    return Project("p", store_root=tmp_path / "store")


# ---------------------------------------------------------------------------
# execute: dependency seeding, graph, rerun policy
# ---------------------------------------------------------------------------

def test_execute_seeds_deps_and_records_graph(mongo, fixed_tree, tmp_path):
    execute(
        "run", {"value": 3}, "run", {"run": _Kernel},
        project=_project(tmp_path), snapshot=False,
    )

    doc = mongo["p"]["runs"].docs[0]
    provenance_record = doc["provenance"]
    dep_names = {d["name"] for d in provenance_record["code"]["deps"]}
    assert _Kernel.__module__ in dep_names

    graph = provenance_record["code"]["graph"]
    assert graph["run"]["impl"].endswith(":_Kernel")
    assert graph["run"]["deps"] == ["value"]
    assert doc["metrics"]["value"] == 3


def test_rerun_missing_skips_and_stale_reruns_on_dep_change(
    mongo, fixed_tree, tmp_path
):
    dep_file = tmp_path / "dep.py"
    dep_file.write_text("VALUE = 1\n", encoding="utf-8")
    project = _project(tmp_path)
    kwargs = dict(
        name="run", spec={"value": 3}, entry="run", factories={"run": _Kernel},
        project=project, deps=[str(dep_file)], snapshot=False,
    )

    execute(**kwargs)
    assert len(mongo["p"]["runs"].docs) == 1

    execute(**kwargs, rerun="missing")
    assert len(mongo["p"]["runs"].docs) == 1  # completed spec is skipped

    dep_file.write_text("VALUE = 2\n", encoding="utf-8")
    execute(**kwargs, rerun="stale")
    assert len(mongo["p"]["runs"].docs) == 2  # changed dependency ran again

    execute(**kwargs, rerun="stale")
    assert len(mongo["p"]["runs"].docs) == 2  # now current again

    execute(**kwargs, rerun="always")
    assert len(mongo["p"]["runs"].docs) == 2  # replaces its own document


def test_run_sweep_aborts_when_project_is_frozen(mongo, fixed_tree, tmp_path):
    project = _project(tmp_path)
    project._manifest = {"_kind": "tracking.manifest", "frozen": True}
    project._manifest_at = 1e18

    sweep = Sweep(
        cases=[{"value": 1}, {"value": 2}],
        factories={"run": _Kernel},
        project=project,
        snapshot=False,
    )

    with pytest.raises(FrozenProjectError):
        run_sweep("run", sweep)
    assert mongo["p"]["runs"].docs == []


# ---------------------------------------------------------------------------
# status / validity
# ---------------------------------------------------------------------------

def _doc(kind: str, *, env_hash: str, dep_hash: str, inputs=None) -> dict:
    provenance_record = {
        "schema_version": 1,
        "code": {
            "git": {"commit": "abc", "branch": "main", "dirty": False, "tag": None},
            "tree_hash": "t",
            "deps": [{"name": "tracking.provenance", "sha256": dep_hash}],
            "deps_hash": "d",
            "snapshot": {"path": "trees/t.tar.gz", "sha256": "s"},
            "graph": {},
        },
        "env": {"python": "3.12", "packages": {}, "hash": env_hash},
        "inputs": inputs or [],
        "inputs_hash": "i",
        "verified": True,
    }
    doc = {
        "meta": {"run_name": kind},
        "spec_hash": kind,
        "run_key": kind,
        "tier": "draft",
        "dimensions": {"x": 1},
        "metrics": {},
        "status": "COMPLETED",
        "invalidated": None,
        "provenance": provenance_record,
    }
    if kind == "UNKNOWN":
        doc.pop("provenance")
    if kind == "INVALIDATED":
        doc["invalidated"] = {"at": None, "reason": "chunking fix"}
    return doc


def test_status_classifies_every_validity_state(mongo, fixed_tree, tmp_path):
    current_env = provenance.env_fingerprint()["hash"]
    current_dep = provenance.resolve_dep("tracking.provenance")["sha256"]

    collection = mongo["p"]["runs"]
    collection.docs = [
        _doc("CURRENT", env_hash=current_env, dep_hash=current_dep),
        _doc("STALE_CODE", env_hash=current_env, dep_hash="0" * 64),
        _doc("STALE_ENV", env_hash="0" * 64, dep_hash=current_dep),
        _doc(
            "STALE_INPUT",
            env_hash=current_env,
            dep_hash=current_dep,
            inputs=[{"name": "in.txt", "path": "", "sha256": "f" * 64,
                     "object": "objects/ff/" + "f" * 64}],
        ),
        _doc("INVALIDATED", env_hash=current_env, dep_hash=current_dep),
        _doc("UNKNOWN", env_hash=current_env, dep_hash=current_dep),
    ]

    status = _project(tmp_path).status("runs").set_index("meta_run_name")
    assert status.loc["CURRENT", "validity"] == "CURRENT"
    assert status.loc["STALE_CODE", "validity"] == "STALE_CODE"
    assert "tracking.provenance" in status.loc["STALE_CODE", "validity_detail"]
    assert status.loc["STALE_ENV", "validity"] == "STALE_ENV"
    assert status.loc["STALE_INPUT", "validity"] == "STALE_INPUT"
    assert status.loc["INVALIDATED", "validity"] == "INVALIDATED"
    assert status.loc["INVALIDATED", "validity_detail"] == "chunking fix"
    assert status.loc["UNKNOWN", "validity"] == "UNKNOWN"


def test_invalidate_marks_without_deleting(mongo, fixed_tree, tmp_path):
    collection = mongo["p"]["runs"]
    collection.docs = [
        _doc("A", env_hash="e", dep_hash="d"),
        _doc("B", env_hash="e", dep_hash="d"),
    ]

    modified = _project(tmp_path).invalidate(
        "runs", {"meta.run_name": "A"}, reason="bad input"
    )

    assert modified == 1
    assert len(collection.docs) == 2
    assert collection.docs[0]["invalidated"]["reason"] == "bad input"
    assert collection.docs[1]["invalidated"] is None
