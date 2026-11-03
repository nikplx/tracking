"""Artifact descriptors: blobs stored by content hash, referenced from a run.

Big payloads (matrices, Wannier90 files, profiler reports) never belong in a
Mongo document; they go through a :class:`~tracking.store.Store` and only a
small descriptor — name, hash, size, a stable locator — is embedded in the
run doc's ``artifacts`` list. See :meth:`tracking.experiment.Experiment.add_artifact`
and :meth:`tracking.experiment.Experiment.add_array`.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from tracking.store import Store


def build_descriptor(
    name: str,
    data: bytes,
    store: Store,
    *,
    media_type: str | None = None,
    format: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sha256 = store.put_bytes(data)
    return {
        "name": name,
        "sha256": sha256,
        "size": len(data),
        "media_type": media_type,
        "format": format,
        "meta": meta or {},
        "uri": store.uri(sha256),
    }


def serialize_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def array_meta(arrays: dict[str, np.ndarray], axes: Any = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }
    }
    if axes is not None:
        meta["axes"] = axes
    return meta
