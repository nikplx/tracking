from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional

import certifi
import numpy as np
import pymongo
import yaml

from tracking._vendor import sacred
from tracking._vendor.sacred.metrics_logger import linearize_metrics
from tracking._vendor.sacred.observers.file_storage import FileStorageObserver
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tracking import observe
from tracking.hashing import (
    _cleaned,
    _successful_file_hashes,
    spec_hash,
    to_bsonable,
)
from tracking.provenance import (
    collect_inputs,
    deps_digest,
    env_fingerprint,
    inputs_digest,
    make_run_key,
    resolve_dep,
    tracking_version,
    tree_fingerprint,
)
from tracking.project import FrozenProjectError, Project, default_project

logger = logging.getLogger()

type Backend = Literal["file", "mongo_compact"]

MONGO_DB = os.environ.get("TRACKING_MONGO_DB", "tracking")
SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 1
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
SACRED_EXPERIMENT = "benchmark"

ex = sacred.Experiment(SACRED_EXPERIMENT, save_git_info=False)


@ex.main
def _main():
    """Dummy command: runs are driven manually via :class:`Experiment`."""
    return None


# ---------------------------------------------------------------------------
# MongoDB plumbing with retries for flaky HPC networks
# ---------------------------------------------------------------------------

_MONGO_CLIENTS: dict[str, pymongo.MongoClient] = {}


def get_mongo_url() -> str:
    url = os.environ.get("MONGODB_URL") or os.environ.get("MONGO_URL")
    if url:
        return url
    raise ValueError("MongoDB URL not found. Set the MONGODB_URL or MONGO_URL environment variable.")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
def _connect(url: str) -> pymongo.MongoClient:
    client = pymongo.MongoClient(url, tlsCAFile=certifi.where())
    client.admin.command("ping")
    return client


def get_mongo_client(uri: str | None = None) -> pymongo.MongoClient:
    """Return a cached client for ``uri`` (or the environment's URL).

    One client per URL, so projects that share a cluster share a connection
    pool while projects on different clusters do not collide.
    """
    url = uri or get_mongo_url()
    client = _MONGO_CLIENTS.get(url)
    if client is None:
        client = _connect(url)
        _MONGO_CLIENTS[url] = client
    return client


def get_runs_collection(
    collection: str,
    db: str | None = None,
    project: Project | None = None,
):
    """Resolve a collection, optionally overriding the database.

    Kept as a module-level function (rather than a ``Project`` method) so the
    existing notebooks and tests that monkeypatch it keep working.
    """
    project = project or default_project()
    if db is not None and db != project.db:
        project = replace(project, db=db)
    return project.collection(collection)


def _jsonl_path(file_path=None) -> Path:
    return Path(file_path) if file_path is not None else DATA_DIR / "experiments.jsonl"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """One experiment calculation, tracked with Sacred or the compact backend.

    ``backend="file"`` drives a Sacred run with a
    :class:`~tracking._vendor.sacred.observers.FileStorageObserver`;
    ``backend="mongo_compact"`` (the default) writes one self-contained
    MongoDB document per run::

        with Experiment(name, params, backend="mongo_compact",
                        collection="sparsity_v1") as run:
            ...
            run.log_metric("energy", value)
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Any],
        backend: Backend = "mongo_compact",
        checkpoints: bool = True,
        collection: Optional[str] = None,
        project: Optional[Project] = None,
        deps: Optional[list[str]] = None,
        inputs: Optional[list[str | Path]] = None,
        snapshot: bool = True,
    ):
        self.name = name
        self.params = params
        if backend not in ("file", "mongo_compact"):
            raise ValueError(
                f"unknown backend {backend!r}; expected 'mongo_compact' or 'file'"
            )
        self.backend = backend
        self.checkpoint_enabled = checkpoints
        self.collection = collection or "runs"
        self._project = project
        self.deps = list(deps or [])
        self.inputs = list(inputs or [])
        self.snapshot = snapshot
        self.tier = "draft"
        self.timestamp: Optional[datetime] = None
        self.duration = 0.0
        self.metrics: Dict[str, Any] = {}
        self.checkpoint_log: list[dict] = []
        self._sacred_run = None
        self._observe_token = None
        self._scalar_steps: Dict[str, int] = {}
        self._provenance: Optional[dict] = None
        self._run_key: Optional[str] = None
        self._tree: Optional[dict] = None
        self._provenance_root = Path.cwd()

    # -- project / provenance helpers ------------------------------------
    @property
    def project(self) -> Project:
        return self._project or default_project()

    @property
    def run_key(self) -> Optional[str]:
        """Identity of this computation; available after :meth:`prepare`."""
        return self._run_key

    def _collection(self):
        """Resolve the target collection.

        The no-project path calls the module-level helper with the same
        single argument it always took, so monkeypatching
        ``tracking.experiment.get_runs_collection`` keeps working.
        """
        if self._project is None:
            return get_runs_collection(self.collection)
        return get_runs_collection(self.collection, project=self._project)

    def _tree_excludes(self) -> tuple[str, ...]:
        excludes = tuple(self.project.tree_excludes)
        try:
            store_rel = self.project.store.root.resolve().relative_to(self._provenance_root)
            excludes += (store_rel.as_posix(),)
        except ValueError:
            pass
        return excludes

    def prepare(self) -> None:
        """Capture provenance and the run key without starting the run.

        Idempotent. Called by ``__enter__`` and by the driver before its
        dedup check, so ``rerun="stale"`` can compare the *current*
        computation identity against stored ones without duplicating work.
        """
        if self._provenance is not None:
            return
        self._provenance = self._capture_provenance()
        self._run_key = make_run_key(
            spec_hash(self.params),
            self._provenance["code"]["deps_hash"],
            self._provenance["inputs_hash"],
        )

    def _capture_provenance(self) -> dict:
        tree = tree_fingerprint(self._provenance_root, excludes=self._tree_excludes())
        self._tree = tree
        deps = sorted((resolve_dep(d) for d in self.deps), key=lambda d: d["name"])
        inputs = collect_inputs(self.inputs, self.project.store)
        env = env_fingerprint()
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "code": {
                "git": {k: tree[k] for k in ("commit", "branch", "dirty", "tag")},
                "tree_hash": tree["tree_hash"],
                "deps": deps,
                "deps_hash": deps_digest(deps),
                "snapshot": None,
                "graph": {},
            },
            "env": env,
            "inputs": inputs,
            "inputs_hash": inputs_digest(inputs),
            "verified": True,
        }

    def set_graph(self, graph: dict) -> None:
        """Attach the declared computation graph (from :class:`Resolver`)."""
        if self._provenance is not None:
            self._provenance["code"]["graph"] = graph

    def _snapshot_source_tree(self, success: bool) -> None:
        """Capture a source tarball on success.

        Archiving must never fail a long calculation: on error we log, leave
        ``snapshot`` as ``None`` and let promotion decide what to do with the
        unverified run.
        """
        if not success or not self.snapshot:
            return
        if self._provenance is None or self._tree is None:
            return
        try:
            self._provenance["code"]["snapshot"] = self.project.store.put_tree(
                self._provenance_root,
                self._tree,
                excludes=self._tree_excludes(),
            )
        except Exception:
            logger.warning("could not snapshot source tree", exc_info=True)

    def _run_fields(self) -> dict:
        return {
            "run_key": self._run_key,
            "tier": self.tier,
            "invalidated": None,
            "promoted_at": None,
            "provenance": self._provenance,
        }

    # -- active-run plumbing (keeps ``benchmark.py`` metric isolation working)
    @staticmethod
    def get_active() -> Optional["Experiment"]:
        return observe.get_active()

    # -- metrics ---------------------------------------------------------
    def log_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value
        if self._sacred_run is not None:
            self._sacred_run.info[key] = to_bsonable(value)
            if isinstance(value, (bool, int, float, np.number)):
                step = self._scalar_steps.get(key, 0) + 1
                self._scalar_steps[key] = step
                self._sacred_run.log_scalar(key, float(value), step)

    def increase_metric(self, key: str, diff: float = 1) -> None:
        self.log_metric(key, self.metrics.get(key, 0) + diff)

    def append_metric(self, key: str, value: Any) -> None:
        self.metrics.setdefault(key, []).append(value)
        if self._sacred_run is not None:
            self._sacred_run.info[key] = to_bsonable(self.metrics[key])

    def checkpoint(self, name: str) -> None:
        if not self.checkpoint_enabled:
            return
        now = datetime.now()
        prev = self.checkpoint_log[-1]["timestamp"] if self.checkpoint_log else self.timestamp
        diff = (now - prev).total_seconds() if prev is not None else 0.0
        self.checkpoint_log.append({"name": name, "timestamp": now, "diff": diff})
        if self.backend == "mongo_compact":
            self._compact_upsert()
            return
        if self._sacred_run is not None:
            self._sacred_run.info["checkpoints"] = [
                {"name": c["name"], "timestamp": str(c["timestamp"]), "diff": c["diff"]}
                for c in self.checkpoint_log
            ]
            self._flush()  # phase boundary: persist progress for long HPC runs

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.log_metric(f"{name}_time", time.perf_counter() - t0)

    def add_profile(self, prof) -> None:
        """Store a pyinstrument profiler report as a Sacred artifact."""
        html = prof.output_html()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", prefix="profile_", delete=False
        ) as f:
            f.write(html)
            path = f.name
        try:
            if self._sacred_run is not None:
                self._sacred_run.add_artifact(path, name="profile.html")
            else:
                logger.warning("No active Sacred run; profile artifact dropped.")
        finally:
            os.unlink(path)

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "Experiment":
        self.timestamp = datetime.now()
        self.start_time = time.perf_counter()
        self.checkpoint_log = []
        logger.info(
            "starting benchmark run %s.%s",
            self.collection,
            self.name,
        )

        config = _cleaned(self.params)
        logger.info("running experiment with config:\n" + yaml.dump(config))

        self.prepare()

        if self.backend == "mongo_compact":
            self._observe_token = observe.set_active(self)
            self._compact_upsert(status="RUNNING")
            return self

        config["benchmark_name"] = self.name
        # Dedup key: lets Experiment.exists find this run with a single
        # indexed lookup instead of one equality clause per dimension.
        config["spec_hash"] = spec_hash(self.params)
        observers = [FileStorageObserver(DATA_DIR / "sacred_runs")]
        previous, ex.observers = ex.observers, observers
        try:
            # ``force``: benchmark params are dynamic (one grid point's spec
            # differs from another's), so they are never pre-declared in an
            # ``@ex.config`` function.
            self._sacred_run = ex._create_run(
                config_updates=config, info={}, options={"--force": True}
            )
        finally:
            ex.observers = previous
        run = self._sacred_run
        run.observers = sorted(run.observers, key=lambda o: -o.priority)
        # NOTE: we drive the observer lifecycle manually (no ``run()`` call),
        # so there is no stdout-capture context and no heartbeat thread.
        # ``_flush`` below replicates what a heartbeat persists.
        run._emit_started()

        self._observe_token = observe.set_active(self)
        return self

    def _flush(self) -> None:
        """Persist current ``info``/scalars, like a Sacred heartbeat would."""
        run = self._sacred_run
        if run is None:
            return
        beat_time = datetime.now(timezone.utc)
        metrics_by_name = linearize_metrics(run._metrics.get_last_metrics())
        for observer in run.observers:
            run._safe_call(
                observer, "log_metrics", metrics_by_name=metrics_by_name, info=run.info
            )
            run._safe_call(
                observer,
                "heartbeat_event",
                info=run.info,
                captured_out=run.captured_out,
                beat_time=beat_time,
                result=run.result,
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.duration = time.perf_counter() - self.start_time
        self.log_metric("total_duration", self.duration)
        if exc_type:
            self.metrics["status"] = "FAILED"
            logger.error("benchmark run failed: %s", exc_val)
        else:
            self.metrics["status"] = "SUCCESS"

        self._snapshot_source_tree(success=exc_type is None)

        if self.backend == "mongo_compact":
            self._compact_upsert(
                status="FAILED" if exc_type else "COMPLETED")
            if self._observe_token is not None:
                observe.reset(self._observe_token)
                self._observe_token = None
            return False  # never suppress exceptions

        run = self._sacred_run
        if run is not None:
            # Final flush: authoritative state is ``self.metrics`` (a caller
            # composing sub-runs, e.g. via ``observe.isolated()``, may mutate
            # it directly to fold in per-sub-run results).
            info = {k: to_bsonable(v) for k, v in self.metrics.items()}
            info["checkpoints"] = [
                {"name": c["name"], "timestamp": str(c["timestamp"]), "diff": c["diff"]}
                for c in self.checkpoint_log
            ]
            run.info.clear()
            run.info.update(info)
            try:
                self._flush()  # persist final info/metrics (heartbeat payload)
                if exc_type:
                    run._emit_failed(exc_type, exc_val, exc_tb)
                else:
                    run._stop_time()
                    run._emit_completed(None)
            finally:
                run._wait_for_observers()

        if self.backend == "file":
            self._save_to_jsonl()

        if self._observe_token is not None:
            observe.reset(self._observe_token)
            self._observe_token = None
        self._sacred_run = None
        return False  # never suppress exceptions

    # -- dedup -----------------------------------------------------------
    @classmethod
    def exists(
        cls,
        params: Dict[str, Any],
        backend: Backend = "mongo_compact",
        collection: Optional[str] = None,
        project: Optional[Project] = None,
        rerun: Literal["missing", "stale", "always"] = "missing",
        run_key: Optional[str] = None,
    ) -> bool:
        """Skip check: has this computation already completed?

        ``rerun="missing"`` (default) matches any completed run with the same
        dimensions, so a code change never triggers recomputation.
        ``rerun="stale"`` additionally requires the *current* computation
        identity (``run_key``) to match, so changed code or inputs do re-run.
        ``rerun="always"`` never skips.

        The file backend has no run-key index; ``"stale"`` degrades to
        ``"missing"`` there.
        """
        if rerun == "always":
            return False
        if rerun not in ("missing", "stale"):
            raise ValueError(f"unknown rerun policy: {rerun!r}")

        h = spec_hash(params)
        if backend == "mongo_compact":
            coll = (
                get_runs_collection(collection or "runs")
                if project is None
                else get_runs_collection(collection or "runs", project=project)
            )
            try:
                coll.create_index([("spec_hash", 1), ("run_key", 1), ("status", 1)])
            except Exception:
                logger.debug("could not create spec_hash index", exc_info=True)
            query: dict[str, Any] = {"spec_hash": h, "status": "COMPLETED"}
            if rerun == "stale":
                if run_key is None:
                    raise ValueError("rerun='stale' requires a run_key")
                query["run_key"] = run_key
            return coll.find_one(query, {"_id": 1}) is not None
        elif backend == "file":
            if rerun == "stale":
                logger.debug("file backend has no run-key index; treating 'stale' as 'missing'")
            return h in _successful_file_hashes(_jsonl_path())
        else:
            raise ValueError(f"unknown backend: {backend!r}")

    # -- compact mongo backend: one self-contained doc per run -----------
    def _compact_doc(self, status: str) -> Dict[str, Any]:
        """Same shape as a :meth:`_save_to_jsonl` row, plus query helpers."""
        doc = {
            "meta": {
                "run_name": self.name,
                "timestamp": str(self.timestamp),
                "tracking_version": tracking_version(),
                "schema_version": SCHEMA_VERSION,
            },
            "spec_hash": spec_hash(self.params),
            "dimensions": _cleaned(self.params),
            "metrics": {k: to_bsonable(v) for k, v in self.metrics.items()},
            "checkpoints": [
                {"name": c["name"], "timestamp": str(c["timestamp"]), "diff": c["diff"]}
                for c in self.checkpoint_log
            ],
            "status": status,
            "updated": datetime.now(timezone.utc),
        }
        doc.update(self._run_fields())
        return doc

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=30),
        retry=retry_if_not_exception_type(FrozenProjectError),
    )
    def _compact_upsert(self, status: str | None = None) -> None:
        """Insert-or-replace this run's doc, keyed by ``spec_hash`` + ``run_key``.

        The pair matters: re-running the *same* computation (same dimensions,
        code and inputs) replaces its own document, while re-running after a
        code or input change inserts alongside the previous result instead of
        destroying it. Persists on entry (``RUNNING``), every
        :meth:`checkpoint`, and exit (``COMPLETED``/``FAILED``).
        """
        if self._project is not None:
            self._project.assert_writable()
        st = status or self.metrics.get("status") or "RUNNING"
        doc = self._compact_doc(st)
        doc["metrics"]["status"] = self.metrics.get("status", st)
        coll = self._collection()
        flt = {"spec_hash": doc["spec_hash"]}
        if doc.get("run_key"):
            flt["run_key"] = doc["run_key"]
        coll.replace_one(flt, doc, upsert=True)

    # -- file backend ----------------------------------------------------
    def _save_to_jsonl(self, file_path=None) -> None:
        if self._project is not None:
            self._project.assert_writable()
        file_path = _jsonl_path(file_path)
        row = {
            "meta": {
                "run_name": self.name,
                "timestamp": str(self.timestamp),
                "tracking_version": tracking_version(),
                "schema_version": SCHEMA_VERSION,
            },
            "spec_hash": spec_hash(self.params),
            "dimensions": _cleaned(self.params),
            "metrics": {k: to_bsonable(v) for k, v in self.metrics.items()},
            "checkpoints": [
                {"name": c["name"], "timestamp": str(c["timestamp"]), "diff": c["diff"]}
                for c in self.checkpoint_log
            ],
        }
        row.update(self._run_fields())
        with open(file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        logger.info("Run %s saved to JSONL (%s)", self.name, file_path)


# ---------------------------------------------------------------------------
# Reading compact and older Sacred collections with pandas
# ---------------------------------------------------------------------------

def normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one Mongo document into ``dimensions_*`` / ``metrics_*`` columns.

    Handles both the compact shape written by this package (``dimensions`` /
    ``metrics`` / ``meta``) and the older Sacred run-document shape
    (``config`` / ``info`` / ``status``), so collections written before the
    compact backend still load.
    """
    flat: Dict[str, Any] = {}
    if "_id" in doc:
        flat["_id"] = str(doc["_id"])
    if "config" in doc:  # older Sacred shape
        for k, v in (doc.get("config") or {}).items():
            flat[f"dimensions_{k}"] = v
        for k, v in (doc.get("info") or {}).items():
            if k == "checkpoints":
                flat["checkpoints"] = v
            elif k == "metrics":
                continue  # Sacred-internal scalar time-series references
            else:
                flat[f"metrics_{k}"] = v
        flat["metrics_status"] = doc.get("status")
        if doc.get("start_time") is not None:
            flat["meta_timestamp"] = doc["start_time"]
        exp = doc.get("experiment") or {}
        if exp.get("name"):
            flat["meta_run_name"] = exp["name"]
        host = doc.get("host_info") or doc.get("host") or {}
        if host.get("hostname"):
            flat["meta_hostname"] = host["hostname"]
    else:  # compact shape
        for section, prefix in (("meta", "meta"), ("dimensions", "dimensions"), ("metrics", "metrics")):
            for k, v in (doc.get(section) or {}).items():
                flat[f"{prefix}_{k}"] = v
        if "checkpoints" in doc:
            flat["checkpoints"] = doc["checkpoints"]
        if "spec_hash" in doc:
            flat["spec_hash"] = doc["spec_hash"]
        if "run_key" in doc:
            flat["run_key"] = doc["run_key"]
        if "tier" in doc:
            flat["tier"] = doc["tier"]

        invalidated = doc.get("invalidated")
        flat["invalidated"] = bool(invalidated)
        if isinstance(invalidated, dict):
            flat["invalidated_at"] = invalidated.get("at")
            flat["invalidated_reason"] = invalidated.get("reason")

        provenance = doc.get("provenance") or {}
        if provenance:
            code = provenance.get("code") or {}
            git = code.get("git") or {}
            flat["provenance_schema_version"] = provenance.get("schema_version")
            flat["provenance_verified"] = provenance.get("verified")
            flat["provenance_code_git_commit"] = git.get("commit")
            flat["provenance_code_git_dirty"] = git.get("dirty")
            flat["provenance_code_tree_hash"] = code.get("tree_hash")
            flat["provenance_code_deps_hash"] = code.get("deps_hash")
            flat["provenance_env_hash"] = (provenance.get("env") or {}).get("hash")
            flat["provenance_inputs_hash"] = provenance.get("inputs_hash")
    return flat

