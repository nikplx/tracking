import inspect

import pytest

import tracking.experiment as experiment
from tracking import Experiment, observe
from tracking.driver import Sweep, execute


class _FakeCollection:
    """Minimal in-memory stand-in for a pymongo collection."""

    def __init__(self):
        self.docs = {}

    def create_index(self, *args, **kwargs):
        return "spec_hash_1_status_1"

    def find_one(self, query, projection=None):
        for doc in self.docs.values():
            if all(doc.get(key) == value for key, value in query.items()):
                return {"_id": "fake"}
        return None

    def replace_one(self, filt, doc, upsert=False):
        self.docs[filt["spec_hash"]] = doc


def test_default_backend_is_compact_mongo():
    assert inspect.signature(Experiment).parameters["backend"].default == "mongo_compact"
    assert inspect.signature(execute).parameters["backend"].default == "mongo_compact"
    assert Sweep(cases=[], factories={}).backend == "mongo_compact"


@pytest.mark.parametrize("backend", ["mongodb", "file", "unknown"])
def test_removed_or_unknown_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="unknown backend"):
        Experiment("run", {}, backend=backend)


def test_compact_backend_roundtrip(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(experiment, "get_runs_collection", lambda name: collection)

    with Experiment(
        "run", {"x": 1}, backend="mongo_compact", collection="runs"
    ) as run:
        observe.log_metric("energy", 2.5)
        observe.checkpoint("done")

    assert Experiment.exists(
        {"x": 1}, backend="mongo_compact", collection="runs"
    )

    doc = next(iter(collection.docs.values()))
    assert doc["dimensions"] == {"x": 1}
    assert doc["metrics"]["energy"] == 2.5
    assert doc["metrics"]["status"] == "SUCCESS"
    assert doc["status"] == "COMPLETED"
    assert doc["checkpoints"][0]["name"] == "done"
    assert doc["meta"]["run_name"] == "run"
    assert "updated" in doc
