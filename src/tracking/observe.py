from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class RunTracer(Protocol):
    """Minimal interface a benchmark run must implement."""

    def log_metric(self, key: str, value: Any) -> None: ...
    def checkpoint(self, name: str) -> None: ...
    def append_metric(self, key: str, value: Any) -> None:
        """Append ``value`` to the list metric stored under ``key``."""
        ...

    def add_artifact(
        self,
        name: str,
        data: Any,
        *,
        media_type: str | None = None,
        format: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def add_array(
        self, name: str, arrays: dict[str, Any], *, axes: Any = None
    ) -> dict[str, Any]: ...


class NullTracer:
    """No-op tracer used when observability is disabled."""

    def log_metric(self, key: str, value: Any) -> None:
        return None

    def checkpoint(self, name: str) -> None:
        return None

    def append_metric(self, key: str, value: Any) -> None:
        return None

    def add_artifact(self, name: str, data: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def add_array(self, name: str, arrays: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {}


_current: contextvars.ContextVar[RunTracer | None] = contextvars.ContextVar(
    "active_run", default=None
)


def get_active() -> RunTracer | None:
    """Return the currently active run tracer, or ``None``."""
    return _current.get()


def set_active(run: RunTracer | None):
    """Install ``run`` as the active tracer; returns a token for :func:`reset`."""
    return _current.set(run)


def reset(token) -> None:
    """Restore the tracer active before the matching :func:`set_active` call."""
    _current.reset(token)


def log_metric(key: str, value: Any) -> None:
    """Log ``value`` under ``key`` on the active run, if any."""
    run = _current.get()
    if run is not None:
        run.log_metric(key, value)


def checkpoint(name: str) -> None:
    """Record a named checkpoint on the active run, if any."""
    run = _current.get()
    if run is not None:
        run.checkpoint(name)


def append_metric(key: str, value: Any) -> None:
    """Append ``value`` to the list metric ``key`` on the active run, if any."""
    run = _current.get()
    if run is not None:
        run.append_metric(key, value)


def add_artifact(
    name: str,
    data: Any,
    *,
    media_type: str | None = None,
    format: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store ``data`` as an artifact of the active run, if any."""
    run = _current.get()
    if run is None:
        return {}
    return run.add_artifact(name, data, media_type=media_type, format=format, meta=meta)


def add_array(name: str, arrays: dict[str, Any], *, axes: Any = None) -> dict[str, Any]:
    """Store ``arrays`` as a ``.npz`` artifact of the active run, if any."""
    run = _current.get()
    if run is None:
        return {}
    return run.add_array(name, arrays, axes=axes)


@contextmanager
def measure(name: str) -> Iterator[None]:
    """Measure a block of code as ``<name>_time`` on the active run, if any."""
    run = _current.get()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if run is not None:
            run.log_metric(f"{name}_time", time.perf_counter() - t0)


class _Capture:
    """A :class:`RunTracer` that stores everything in a private dict instead
    of reaching whatever backend the real active run is using."""

    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}
        self.artifacts: list[dict[str, Any]] = []

    def log_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def checkpoint(self, name: str) -> None:
        return None

    def append_metric(self, key: str, value: Any) -> None:
        self.metrics.setdefault(key, []).append(value)

    def add_artifact(self, name: str, data: Any, **kwargs: Any) -> dict[str, Any]:
        # No project/store reference here (see module docstring): captured
        # for the caller to inspect and re-add via the real active run,
        # rather than silently dropped or written outside the caller's
        # control.
        record = {"name": name, "data": data, **kwargs}
        self.artifacts.append(record)
        return record

    def add_array(self, name: str, arrays: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.add_artifact(name, arrays, **kwargs)


@contextmanager
def isolated() -> Iterator[dict[str, Any]]:
    """Run a block with metrics captured into a private dict instead of the
    active run's backend, and hand that dict back to the caller.

    This is how a run composed of several parts (a reaction over several
    molecules, a trajectory over several frames, a path over several images)
    runs each part through the exact same tracked-calculation machinery,
    reads back whatever that part chose to log, and decides for itself
    whether and how to fold it into its own metrics -- without the parts
    stepping on each other's keys, and without this module (or whatever
    drives the outer run) needing to know what kind of calculation any of
    it was.
    """
    capture = _Capture()
    token = set_active(capture)
    try:
        yield capture.metrics
    finally:
        reset(token)
