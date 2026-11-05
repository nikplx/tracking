import inspect
import json

import numpy as np
import pandas as pd
import pytest

import tracking.experiment as experiment
from tracking import Experiment, Project, load_file_data, observe
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


@pytest.mark.parametrize("backend", ["mongodb", "unknown"])
def test_removed_or_unknown_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="unknown backend"):
        Experiment("run", {}, backend=backend)


def _file_project(tmp_path):
    return Project(
        "p",
        backend="file",
        file_root=tmp_path / "runs",
        store_root=tmp_path / "store",
    )


def test_file_backend_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "DATA_DIR", tmp_path)
    project = _file_project(tmp_path)

    with Experiment("run", {"x": 1}, backend="file", project=project) as run:
        observe.log_metric("energy", 1.5)
        observe.checkpoint("done")
        run.add_artifact("notes", b"hello", media_type="text/plain")

    assert Experiment.exists({"x": 1}, backend="file", project=project)

    files = list((tmp_path / "runs" / "runs").glob("*.json"))
    assert len(files) == 1
    assert files[0].suffix == ".json"
    row = json.loads(files[0].read_text(encoding="utf-8"))
    assert row["dimensions"] == {"x": 1}
    assert row["metrics"]["energy"] == 1.5
    assert row["metrics"]["status"] == "SUCCESS"
    assert row["checkpoints"][0]["name"] == "done"
    assert row["status"] == "COMPLETED"

    df = load_file_data(project.file_collection("runs"))
    assert df.loc[0, "metrics_energy"] == 1.5

    root_df = load_file_data(project.file_root)
    assert root_df.loc[0, "metrics_energy"] == 1.5

    from tracking import load_json_data

    json_df = load_json_data(project.file_root)
    assert json_df.loc[0, "metrics_energy"] == 1.5

    loaded = project.load("runs")
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.loc[0, "metrics_energy"] == 1.5

    artifacts = project.load_artifacts("runs")
    assert len(artifacts) == 1
    assert artifacts.loc[0, "filename"] == "notes.txt"
    assert artifacts.loc[0, "path"].endswith(".txt")


def test_file_backend_project_artifacts_materialize_with_endings(tmp_path):
    project = _file_project(tmp_path)

    def body(run):
        run.add_array("density", {"dm": np.array([1, 2, 3])})

    run = Experiment("run", {"x": 1}, backend="file", project=project)
    run._provenance_root = tmp_path
    with run:
        body(run)

    out = tmp_path / "export"
    df = project.load_artifacts("runs", dest=out)
    path = tmp_path / "export" / df.loc[0, "filename"]
    assert path.suffix == ".npz"
    assert path.is_file()
    assert df.loc[0, "path"] == str(path)


def test_file_backend_status_and_invalidate(tmp_path):
    project = _file_project(tmp_path)
    run = Experiment("run", {"x": 1}, backend="file", project=project)
    run._provenance_root = tmp_path
    with run:
        observe.log_metric("energy", 1.0)

    status = project.status("runs")
    assert len(status) == 1
    assert "validity" in status.columns

    modified = project.invalidate(
        "runs", {"meta.run_name": "run"}, reason="bad input"
    )
    assert modified == 1

    status = project.status("runs")
    assert status.loc[0, "validity"] == "INVALIDATED"
    assert status.loc[0, "validity_detail"] == "bad input"


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
