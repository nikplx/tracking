"""Safe system/host metadata for a run's provenance record.

This is a *debugging aid*, not an identity: it travels under
``provenance.system`` (see :func:`tracking.experiment.Experiment._capture_provenance`)
and is never part of ``run_key``/``spec_hash``, so two runs on different
hosts with identical code/inputs still dedupe against each other.

Every probe is individually best-effort — a missing ``/proc/meminfo`` on a
non-Linux box, or a locked-down ``platform`` call, drops just that one field
rather than the whole fingerprint. Only an explicit allowlist of values is
collected (no raw ``os.environ`` dump), and everything returned is a BSON-safe
scalar. Set ``TRACKING_SYSTEM_INFO=0`` to skip this capture entirely.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SLURM_FIELDS = {
    "job_id": "SLURM_JOB_ID",
    "array_job_id": "SLURM_ARRAY_JOB_ID",
    "array_task_id": "SLURM_ARRAY_TASK_ID",
    "nodelist": "SLURM_NODELIST",
    "cpus_on_node": "SLURM_CPUS_ON_NODE",
    "partition": "SLURM_JOB_PARTITION",
    "cluster": "SLURM_CLUSTER_NAME",
}


def _safe_set(info: dict[str, Any], key: str, probe: Callable[[], Any]) -> None:
    try:
        value = probe()
    except Exception:
        logger.debug("system fingerprint: probe %r failed", key, exc_info=True)
        return
    if value is not None:
        info[key] = value


def _mem_total_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return round(kib / (1024 * 1024), 2)
    except OSError:
        return None
    return None


def _slurm_fingerprint() -> dict[str, str]:
    if not os.environ.get("SLURM_JOB_ID"):
        return {}
    return {
        key: os.environ[var] for key, var in _SLURM_FIELDS.items() if os.environ.get(var)
    }


def _container_fingerprint() -> dict[str, bool]:
    if (
        Path("/.singularity.d").exists()
        or os.environ.get("APPTAINER_CONTAINER")
        or os.environ.get("SINGULARITY_CONTAINER_NAME")
    ):
        return {"apptainer": True}
    return {}


def system_fingerprint() -> dict[str, Any]:
    """An allowlisted, failure-tolerant snapshot of the current host.

    Returns ``{}`` if ``TRACKING_SYSTEM_INFO=0`` is set.
    """
    if os.environ.get("TRACKING_SYSTEM_INFO") == "0":
        return {}

    info: dict[str, Any] = {}
    _safe_set(info, "hostname", socket.gethostname)
    _safe_set(info, "os", platform.system)
    _safe_set(info, "kernel", platform.release)
    _safe_set(info, "arch", platform.machine)
    _safe_set(info, "cpu_model", platform.processor)
    _safe_set(info, "cpu_count", os.cpu_count)
    _safe_set(info, "mem_total_gb", _mem_total_gb)
    _safe_set(info, "python_version", platform.python_version)
    _safe_set(info, "python_implementation", platform.python_implementation)
    info["executable"] = sys.executable
    info["cwd"] = os.getcwd()

    slurm = _slurm_fingerprint()
    if slurm:
        info["slurm"] = slurm
    container = _container_fingerprint()
    if container:
        info["container"] = container
    return info
