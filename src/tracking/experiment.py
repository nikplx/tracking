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
from tracking._vendor.sacred.observers import FileStorageObserver
from tracking._vendor.sacred.observers.mongo import QueuedMongoObserver
from tenacity import retry, stop_after_attempt, wait_exponential

from tracking import observe
from tracking.configurable import Configurable

logger = logging.getLogger()

type Backend = Literal["mongodb", "file"]

MONGO_DB = os.environ.get("TRACKING_MONGO_DB", "tracking")
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
SACRED_EXPERIMENT = "benchmark"

ex = sacred.Experiment(SACRED_EXPERIMENT, save_git_info=False)


@ex.main
def _main():
    """Dummy command: runs are driven manually via :class:`BenchmarkRun`."""
    return None


# ---------------------------------------------------------------------------
# BSON-safe cleaning (replaces the old ``clean_value``)
# ---------------------------------------------------------------------------

_TRUNCATE_AT = 4000


def dims_of(v: Any) -> dict[str, Any] | None:
    """Return ``v.dims()`` if it defines that protocol, else ``None``.

    The ``dims()`` protocol lets a parameter object control how it is
    tracked: instead of falling through to ``str(v)`` (a verbose blob for
    dataclasses carrying large payloads like geometries), the returned
    dict of plain scalar values is stored. Only a ``dict`` return value
    is honored; anything else falls through to the normal handling.
    """
    dims = getattr(v, "dims", None)
    if callable(dims):
        d = dims()
        if isinstance(d, dict):
            return d
    return None


def to_bsonable(v: Any) -> Any:
    """Convert arbitrary benchmark params/metrics into BSON-encodable values."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.number):
        return v.item()
    if isinstance(v, np.ndarray):
        return [to_bsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [to_bsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_bsonable(x) for k, x in v.items()}
    if isinstance(v, (Path,)):
        return str(v)
    d = dims_of(v)
    if d is not None:
        return {str(k): to_bsonable(x) for k, x in d.items()}
    if isinstance(v, Configurable):
        return v.__class__.__name__
    if hasattr(v, "name") and isinstance(v.name, str):
        return v.name
    s = str(v)
    if len(s) > _TRUNCATE_AT:
        s = s[:_TRUNCATE_AT] + "...<truncated>"
    return s


def _cleaned(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean params for storage, expanding top-level ``dims()`` objects.

    A top-level value defining ``dims()`` (see :func:`dims_of`) is replaced
    by its dims entries, so dimensions stay flat scalar columns (e.g.
    ``dataset``/``reaction``/``reaction_file``) instead of one verbose
    stringified blob. The raw object is still available to the
    :class:`~tracking.resolve.Resolver` -- this only affects what is
    tracked. Explicit keys in ``params`` win on collision, processed in
    order.
    """
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (dict, list, tuple)):
            continue
        d = dims_of(v)
        if d is not None:
            for dk, dv in d.items():
                out[str(dk)] = to_bsonable(dv)
    # Explicit scalar keys win over expanded dims on collision: re-apply
    # any plain (non-dims) entries last.
    for k, v in params.items():
        if isinstance(v, (dict, list, tuple)) or dims_of(v) is None:
            out[str(k)] = to_bsonable(v)
    return out


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


# ---------------------------------------------------------------------------
# BenchmarkRun
# ---------------------------------------------------------------------------

class Experiment:
    """One experiment calculation, tracked with Sacred.

    Mirrors the old ``ExperimentRun`` workflow so ``benchmark.py`` barely
    changes::

        with BenchmarkRun(name, params, backend="mongodb",
                          collection="sparsity_v1") as run:
            ...
            run.log_metric("energy", value)
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Any],
        backend: Backend = "mongodb",
        checkpoints: bool = True,
        collection: Optional[str] = None,
    ):
        self.name = name
        self.params = params
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
    def _make_observers(self):
        if self.backend == "mongodb":
            try:
                url = get_mongo_url()
            except ValueError:
                logger.warning(
                    "No MongoDB URL found; falling back to local file observer."
                )
                return [FileStorageObserver(DATA_DIR / "sacred_runs")]
            return [
                QueuedMongoObserver(
                    url=url, db_name=MONGO_DB, collection=self.collection
                )
            ]
        return [FileStorageObserver(DATA_DIR / "sacred_runs")]

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

        config["benchmark_name"] = self.name
        observers = self._make_observers()
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
        backend: Backend = "mongodb",
        collection: Optional[str] = None,
    ) -> bool:
        if backend != "mongodb":
            return False
        coll = get_runs_collection(collection or "runs")
        cleaned = _cleaned(params)
        query = {
            f"config.{k}": v
            for k, v in cleaned.items()
            if k != "benchmark_name"
        }
        query["status"] = "COMPLETED"
        return coll.find_one(query, {"_id": 1}) is not None

    # -- legacy file backend ---------------------------------------------
    def _save_to_jsonl(self, file_path: str = DATA_DIR/"experiments.jsonl") -> None:
        row = {
            "meta": {"run_name": self.name, "timestamp": str(self.timestamp)},
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
# Reading mixed collections (legacy + Sacred docs) with pandas
# ---------------------------------------------------------------------------

def normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one Mongo document into ``dimensions_*`` / ``metrics_*`` columns.

    Handles both the legacy shape (``dimensions`` / ``metrics`` / ``meta``)
    and Sacred run documents (``config`` / ``info`` / ``status``).
    """
    flat: Dict[str, Any] = {}
    if "_id" in doc:
        flat["_id"] = str(doc["_id"])
    if "config" in doc:  # Sacred shape
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
    else:  # legacy shape
        for section, prefix in (("meta", "meta"), ("dimensions", "dimensions"), ("metrics", "metrics")):
            for k, v in (doc.get(section) or {}).items():
                flat[f"{prefix}_{k}"] = v
        if "checkpoints" in doc:
            flat["checkpoints"] = doc["checkpoints"]
    return flat

