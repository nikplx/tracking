import io

import numpy as np

from tracking import Experiment, Project, observe


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


def _project(tmp_path):
    """A :class:`Project` whose Mongo collection is an in-memory fake.

    ``Project.collection`` (not the module-level ``get_runs_collection``) is
    what both ``Experiment`` writes through and ``Project.docs``/``status``/
    ``artifacts`` read through, so overriding it here keeps writer and reader
    on the same fake for every test below.
    """
    project = Project("p", store_root=tmp_path / "store")
    collection = _FakeCollection()
    project.collection = lambda name: collection
    return project, collection


def _run(project, tmp_path, body):
    run = Experiment("run", {"x": 1}, project=project)
    run._provenance_root = tmp_path
    with run:
        return body(run)


def test_add_artifact_round_trips_bytes(tmp_path):
    project, collection = _project(tmp_path)
    descriptor = _run(
        project, tmp_path,
        lambda run: run.add_artifact("notes", b"hello world", media_type="text/plain"),
    )

    assert descriptor["name"] == "notes"
    assert descriptor["size"] == len(b"hello world")
    assert descriptor["media_type"] == "text/plain"

    doc = next(iter(collection.docs.values()))
    assert doc["artifacts"] == [descriptor]

    assert project.load_artifact(descriptor) == b"hello world"
    assert project.load_artifact(descriptor["sha256"]) == b"hello world"

    dest = tmp_path / "out.txt"
    assert project.load_artifact(descriptor, dest=dest) == dest
    assert dest.read_bytes() == b"hello world"


def test_artifact_descriptor_uses_store_relative_object(tmp_path):
    project, _ = _project(tmp_path)
    descriptor = _run(
        project, tmp_path,
        lambda run: run.add_array("density", {"dm": np.eye(2)}),
    )

    sha = descriptor["sha256"]
    assert descriptor["object"] == f"objects/{sha[:2]}/{sha}.npz"
    assert "uri" not in descriptor
    assert not descriptor["object"].startswith("/")


def test_add_array_round_trips_arrays(tmp_path):
    project, _ = _project(tmp_path)
    arrays = {"dm": np.eye(3), "energies": np.array([1.0, 2.0, 3.0])}
    descriptor = _run(
        project, tmp_path,
        lambda run: run.add_array("density", arrays, axes="ao,ao"),
    )

    assert descriptor["format"] == "npz"
    assert descriptor["meta"]["axes"] == "ao,ao"
    assert descriptor["meta"]["arrays"]["dm"] == {"shape": [3, 3], "dtype": "float64"}

    data = project.load_artifact(descriptor)
    loaded = np.load(io.BytesIO(data))
    assert np.array_equal(loaded["dm"], arrays["dm"])
    assert np.array_equal(loaded["energies"], arrays["energies"])


def test_observe_add_artifact_dispatches_to_active_run(tmp_path):
    project, collection = _project(tmp_path)
    descriptor = _run(
        project, tmp_path,
        lambda run: observe.add_artifact("obs", b"via-observe"),
    )

    assert descriptor["name"] == "obs"
    doc = next(iter(collection.docs.values()))
    assert doc["artifacts"][0]["sha256"] == descriptor["sha256"]


def test_observe_add_artifact_is_a_noop_without_an_active_run():
    assert observe.add_artifact("x", b"data") == {}
    assert observe.add_array("x", {"a": np.zeros(1)}) == {}


def test_add_profile_stores_an_html_artifact(tmp_path):
    class _FakeProfiler:
        def output_html(self):
            return "<html>profile</html>"

    project, _ = _project(tmp_path)
    descriptor = _run(project, tmp_path, lambda run: run.add_profile(_FakeProfiler()))

    assert descriptor["media_type"] == "text/html"
    assert project.load_artifact(descriptor) == b"<html>profile</html>"


def test_project_artifacts_returns_one_row_per_artifact(tmp_path):
    project, _ = _project(tmp_path)

    def body(run):
        run.add_artifact("a", b"1")
        run.add_artifact("b", b"22")

    _run(project, tmp_path, body)

    rows = project.artifacts("runs")
    assert len(rows) == 2
    assert set(rows["name"]) == {"a", "b"}
    assert set(rows["collection"]) == {"runs"}
    assert rows["run_key"].notna().all()
