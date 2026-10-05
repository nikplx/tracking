"""Compare every run against a chosen reference/baseline group.

The recurring shape: you sweep several methods (or several variants of one
method) over the same underlying cases, one of those groups is the baseline
you actually care about being close to (a reference method, an unsparsified
run, a tighter-threshold run, ...), and you want every numeric metric -
energy, wall time, iteration count, whatever - expressed as "versus that
baseline" instead of as a raw number, for every row, without re-deriving the
join by hand each time your dimensions/metrics schema changes.

This is deliberately the one general case: `add_reference_diffs` doesn't
know what a "sweep" or a "reaction" is. It just needs a column that names
the group each row belongs to, which value of that column is the baseline,
and which columns identify "the same case" across groups.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_reference_diffs(
    df: pd.DataFrame,
    *,
    group: str,
    reference: Any,
    keys: Sequence[str],
    metrics: Optional[Sequence[str]] = None,
    suffix: str = "_vs_ref",
    keep: Literal["first", "last"] = "first",
    return_reference: bool = False,
) -> pd.DataFrame:
    missing_cols = [c for c in [group, *keys] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"column(s) not in dataframe: {missing_cols}")

    ref = df[df[group] == reference]
    if ref.empty:
        raise ValueError(f"no rows with {group!r} == {reference!r}")

    dup = ref.duplicated(subset=list(keys), keep=False)
    if dup.any():
        logger.warning(
            "%d reference rows share a key in %s; keeping the %s of each "
            "duplicate group (keep=%r). Add a column to `keys` to "
            "disambiguate them if they're not really the same case.",
            int(dup.sum()), list(keys), keep, keep,
        )
    ref = ref.drop_duplicates(subset=list(keys), keep=keep)

    if metrics is None:
        reserved = {group, *keys, "is_reference"}
        # Exclude columns a previous call already derived (``_ref``,
        # ``{suffix}``, ``{suffix}_pct``), so auto-discovery stays sane if
        # this is called more than once - e.g. against a second reference.
        derived_endings = (f"{suffix}_pct", suffix, "_ref")
        candidates = [
            c for c in ref.columns
            if c not in reserved
            and not c.endswith(derived_endings)
            and (pd.api.types.is_numeric_dtype(ref[c]) or pd.api.types.is_bool_dtype(ref[c]))
        ]
        # If the frame uses the tracking/normalize_doc "metrics_*" naming
        # convention, default to just those - a numeric *dimension* (a
        # method-only knob like a sparsity threshold) isn't something you
        # usually want auto-diffed. Pass `metrics=` explicitly to override.
        metric_prefixed = [c for c in candidates if c.startswith("metrics_")]
        metrics = metric_prefixed or candidates
    metrics = [m for m in metrics if m in ref.columns]
    if not metrics:
        raise ValueError("no numeric metric columns found to diff")

    # Drop columns a previous call already added for these metrics, so
    # re-running this (same notebook cell, or a second reference) overwrites
    # cleanly instead of colliding with them during the merge below.
    stale = [c for m in metrics for c in (f"{m}_ref", f"{m}{suffix}", f"{m}{suffix}_pct")]
    out = df.drop(columns=[c for c in stale if c in df.columns])

    ref_small = ref[[*keys, *metrics]].rename(columns={m: f"{m}_ref" for m in metrics})
    out = out.merge(ref_small, on=list(keys), how="left")
    out["is_reference"] = out[group] == reference

    n_other = int((~out["is_reference"]).sum())
    worst_unmatched = 0
    for metric in metrics:
        ref_col = f"{metric}_ref"
        # Cast to float for the arithmetic: a boolean metric (e.g.
        # "converged") is a legitimate thing to diff (-1/0/1 = flipped to
        # False / unchanged / flipped to True), but numpy/pandas refuse `-`
        # directly on bool dtype, and a merge can also leave a bool column
        # as object dtype wherever an unmatched row introduced a NaN.
        values = out[metric].astype(float)

        ref_values = out[ref_col].astype(float)
        out[f"{metric}{suffix}"] = values - ref_values

        pct = (out[f"{metric}{suffix}"] / ref_values) * 100.0
        out[f"{metric}{suffix}_pct"] = pct.replace([np.inf, -np.inf], np.nan)

        out[f"{metric}{suffix}_abs"] = out[f"{metric}{suffix}"].abs()

        unmatched = int((out[ref_col].isna() & ~out["is_reference"]).sum())
        worst_unmatched = max(worst_unmatched, unmatched)

    if worst_unmatched:
        logger.warning(
            "up to %d/%d non-reference rows had no matching reference case "
            "for key %s (left as NaN in the diff columns).",
            worst_unmatched, n_other, list(keys),
        )

    return out[out['is_reference'] == return_reference]
