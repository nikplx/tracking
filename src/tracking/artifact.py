"""Artifact descriptors: blobs stored by content hash, referenced from a run.

Big payloads (matrices, Wannier90 files, profiler reports) never belong in a
Mongo document; they go through a :class:`~tracking.store.Store` and only a
small descriptor — name, hash, size, and a store-relative object key — is
embedded in the run doc's ``artifacts`` list. See :meth:`tracking.experiment.Experiment.add_artifact`
and :meth:`tracking.experiment.Experiment.add_array`.
"""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Any

import numpy as np

from tracking.store import Store, normalize_suffix


def artifact_suffix(
    name: str, *, format: str | None = None, media_type: str | None = None
) -> str:
    """Best-effort file extension for an artifact.

    An explicit ``format`` wins (``"npz"`` -> ``".npz"``), then an extension
    already present on ``name``, then one guessed from ``media_type``.
    """
    if format:
        return normalize_suffix(format)
    suffix = Path(name).suffix
    if suffix:
        return normalize_suffix(suffix)
    if media_type:
        guessed = mimetypes.guess_extension(media_type.split(";", 1)[0].strip())
        if guessed:
            return guessed
    return ""


def artifact_filename(
    name: str, *, format: str | None = None, media_type: str | None = None
) -> str:
    """A human-friendly filename with the artifact's proper ending."""
    suffix = artifact_suffix(name, format=format, media_type=media_type)
    return name if Path(name).suffix or not suffix else f"{name}{suffix}"


def build_descriptor(
    name: str,
    data: bytes,
    store: Store,
    *,
    media_type: str | None = None,
    format: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suffix = artifact_suffix(name, format=format, media_type=media_type)
    sha256 = store.put_bytes(data, suffix=suffix)
    return {
        "name": name,
        "filename": artifact_filename(name, format=format, media_type=media_type),
        "sha256": sha256,
        "size": len(data),
        "media_type": media_type,
        "format": format,
        "meta": meta or {},
        # Store-relative, so run documents stay valid when the store directory
        # or bucket prefix is copied or moved.
        "object": store.object_key(sha256, suffix=suffix),
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
