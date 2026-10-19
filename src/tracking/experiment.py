from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
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
from tenacity import retry, stop_after_attempt, wait_exponential

from tracking import observe
from tracking.hashing import (
    _cleaned,
    _successful_file_hashes,
    spec_hash,
    to_bsonable,
)

logger = logging.getLogger()

type Backend = Literal["file", "mongo_compact"]

MONGO_DB = os.environ.get("TRACKING_MONGO_DB", "tracking")
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

_MONGO_CLIENT: Optional[pymongo.MongoClient] = None


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


def get_mongo_client() -> pymongo.MongoClient:
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        _MONGO_CLIENT = _connect(get_mongo_url())
    return _MONGO_CLIENT


def get_runs_collection(collection: str):
    return get_mongo_client()[MONGO_DB][collection or "runs"]


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
        self.timestamp: Optional[datetime] = None
        self.duration = 0.0
        self.metrics: Dict[str, Any] = {}
        self.checkpoint_log: list[dict] = []
        self._sacred_run = None
        self._observe_token = None
        self._scalar_steps: Dict[str, int] = {}

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
    ) -> bool:
        """Skip check: is there a completed run with these dimensions?

        Compares a single :func:`spec_hash` over the cleaned dimensions
        instead of one equality clause per key. Runs written before the
        hash existed have no ``spec_hash`` and never match (they re-run
        once, then carry the hash going forward).
        """
        h = spec_hash(params)
        if backend == "mongo_compact":
            coll = get_runs_collection(collection or "runs")
            try:
                coll.create_index([("spec_hash", 1), ("status", 1)])
            except Exception:
                logger.debug("could not create spec_hash index", exc_info=True)
            return coll.find_one(
                {"spec_hash": h, "status": "COMPLETED"},
                {"_id": 1},
            ) is not None
        elif backend == "file":
            return h in _successful_file_hashes(_jsonl_path())
        else:
            raise ValueError(f"unknown backend: {backend!r}")

    # -- compact mongo backend: one self-contained doc per run -----------
    def _compact_doc(self, status: str) -> Dict[str, Any]:
        """Same shape as a :meth:`_save_to_jsonl` row, plus query helpers."""
        return {
            "meta": {"run_name": self.name, "timestamp": str(self.timestamp)},
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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def _compact_upsert(self, status: str | None = None) -> None:
        """Insert-or-replace this run's doc, keyed by ``spec_hash``.

        Persists on entry (``RUNNING``), every :meth:`checkpoint`, and exit
        (``COMPLETED``/``FAILED``) — the same phase boundaries at which the
        file backend flushes, so long HPC runs leave resumable state.
        """
        st = status or self.metrics.get("status") or "RUNNING"
        doc = self._compact_doc(st)
        doc["metrics"]["status"] = self.metrics.get("status", st)
        coll = get_runs_collection(self.collection)
        coll.replace_one({"spec_hash": doc["spec_hash"]}, doc, upsert=True)

    # -- file backend ----------------------------------------------------
    def _save_to_jsonl(self, file_path=None) -> None:
        file_path = _jsonl_path(file_path)
        row = {
            "meta": {"run_name": self.name, "timestamp": str(self.timestamp)},
            "spec_hash": spec_hash(self.params),
            "dimensions": _cleaned(self.params),
            "metrics": {k: to_bsonable(v) for k, v in self.metrics.items()},
            "checkpoints": [
                {"name": c["name"], "timestamp": str(c["timestamp"]), "diff": c["diff"]}
                for c in self.checkpoint_log
            ],
        }
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
    return flat

