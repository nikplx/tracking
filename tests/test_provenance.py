import subprocess

from tracking.provenance import (
    collect_inputs,
    deps_digest,
    env_fingerprint,
    make_run_key,
    resolve_dep,
    seed_deps,
    tree_fingerprint,
)
from tracking.store import ContentStore, hash_file


def _git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_resolve_dep_module_and_member():
    module = resolve_dep("tracking.provenance")
    assert module["name"] == "tracking.provenance"
    assert len(module["sha256"]) == 64

    member = resolve_dep("tracking.provenance:make_run_key")
    assert member["name"] == "tracking.provenance:make_run_key"
    assert len(member["sha256"]) == 64
    assert member["sha256"] != module["sha256"]


def test_seed_deps_is_sorted_and_deduplicated():
    from tracking.provenance import make_run_key as a
    from tracking.store import hash_file as b

    assert seed_deps({"x": a, "y": a, "z": b}) == [
        "tracking.provenance",
        "tracking.store",
    ]


def test_env_fingerprint_is_stable_and_complete():
    first = env_fingerprint()
    second = env_fingerprint()
    assert first == second
    assert "python" in first
    assert "numpy" in first["packages"]
    assert len(first["hash"]) == 64


def test_tree_fingerprint_clean_then_dirty(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "init")

    clean = tree_fingerprint(tmp_path)
    assert clean["dirty"] is False
    assert clean["commit"]
    assert clean["files"] == ["a.py"]

    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    dirty = tree_fingerprint(tmp_path)
    assert dirty["dirty"] is True
    assert dirty["tree_hash"] != clean["tree_hash"]
    assert dirty["commit"] == clean["commit"]

    (tmp_path / "b.py").write_text("y = 3\n", encoding="utf-8")
    with_untracked = tree_fingerprint(tmp_path)
    assert with_untracked["files"] == ["a.py", "b.py"]
    assert with_untracked["tree_hash"] != dirty["tree_hash"]


def test_tree_fingerprint_without_git(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    fingerprint = tree_fingerprint(tmp_path)
    assert fingerprint["commit"] is None
    assert fingerprint["dirty"] is None
    assert fingerprint["files"] == ["a.py"]


def test_deps_and_run_key_change_with_code_identity():
    dep_a = {"name": "a", "sha256": "1" * 64}
    dep_b = {"name": "b", "sha256": "2" * 64}
    assert deps_digest([dep_a]) == deps_digest([dep_a])
    assert deps_digest([dep_a]) != deps_digest([dep_b])

    key = make_run_key("spec", deps_digest([dep_a]), "i" * 64)
    assert key == make_run_key("spec", deps_digest([dep_a]), "i" * 64)
    assert key != make_run_key("spec", deps_digest([dep_b]), "i" * 64)


def test_collect_inputs_copies_and_deduplicates(tmp_path):
    store = ContentStore(tmp_path / "store")
    first = tmp_path / "in.txt"
    first.write_text("data", encoding="utf-8")
    second = tmp_path / "copy.txt"
    second.write_text("data", encoding="utf-8")

    records = collect_inputs([first, second], store)
    assert {r["sha256"] for r in records} == {hash_file(first)}
    for record in records:
        assert store.has(record["sha256"])
        assert store.get_bytes(record["sha256"]) == b"data"
    assert records == sorted(records, key=lambda r: (r["name"], r["path"]))
