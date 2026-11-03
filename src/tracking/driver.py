"""The benchmark dispatcher.

This is deliberately small: expand a grid of cases/hyperparameters, and for
each point, skip it if a completed run with this exact spec already exists,
otherwise resolve the benchmark's ``entry`` object from the spec and run it.

It never branches on what kind of calculation is running. What "single
point", "reaction", or (one day) "PBC cell" *means* lives entirely in which
class ``factories[entry]`` points at and what that class's ``kernel()``
does; this module only cares that something ran, once, and got tracked.
"""

from __future__ import annotations

import argparse

import sys

import itertools
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from pyinstrument import Profiler
from tracking import observe
from tracking.experiment import Backend, Experiment
from tracking.provenance import seed_deps
from tracking.project import FrozenProjectError, Project
from tracking.resolve import Factory, Resolver

logger = logging.getLogger(__name__)


@dataclass
class Sweep:
    """One benchmark entry: what varies, and how to wire it up.

    No ``type`` field. Adding a new kind of run (a new system, a new
    method) never touches this dataclass or ``run_sweep`` below -- it's
    purely a new ``factories`` registry pointing ``entry`` at a new
    ``Runnable``.
    """
    cases: Iterable[dict[str, Any]]
    factories: dict[str, Factory]
    entry: str = "run"
    grid: dict[str, list[Any]] = field(default_factory=dict)
    backend: Backend = "mongo_compact"
    collection: str | None = None
    project: Project | None = None
    deps: list[str] | None = None
    inputs: list[str | Path] | None = None
    rerun: Literal["missing", "stale", "always"] = "missing"
    snapshot: bool = True
    checkpoint: bool = False
    profile: bool = False
    overwrite: bool = False


def expand_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys, values = zip(*grid.items())
    for bundle in itertools.product(*values):
        yield dict(zip(keys, bundle))

def execute(
    name: str,
    spec: dict[str, Any],
    entry: str,
    factories: dict[str, Factory],
    *,
    backend: Backend = "mongo_compact",
    collection: str | None = None,
    project: Project | None = None,
    deps: list[str] | None = None,
    inputs: list[str | Path] | None = None,
    rerun: Literal["missing", "stale", "always"] = "missing",
    snapshot: bool = True,
    checkpoint: bool = False,
    profile: bool = False,
    overwrite: bool = False,
) -> None:
    """Run one point unless a completed run with this spec already exists.

    ``spec`` is the raw, pre-resolution config -- file paths, strings,
    numbers. Everything logged and hashed for dedup is plain data, never a
    live resolved object, so there's no risk of dedup silently degrading to
    ``str(some_object)`` because a live object leaked into the tracked
    params.
    What gets built from ``spec``, and what it logs while running, is
    entirely the resolved entry object's business.

    Provenance dependencies are the union of ``deps`` and the modules of
    every registered factory, so a sweep records the code it actually
    depends on without extra bookkeeping.
    """
    dep_specs = list(dict.fromkeys([*(deps or []), *seed_deps(factories)]))
    factory_spec = {**spec, **{f"{k}_factory": v.__name__ for k, v in factories.items()}}

    run = Experiment(
        name=name,
        params=factory_spec,
        backend=backend,
        checkpoints=checkpoint,
        collection=collection,
        project=project,
        deps=dep_specs,
        inputs=inputs,
        snapshot=snapshot,
    )

    if not overwrite and rerun != "always":
        if rerun == "missing":
            skip = Experiment.exists(
                spec, backend=backend, collection=collection,
                project=project, rerun="missing",
            )
        else:  # "stale": compare the current computation identity
            run.prepare()
            skip = Experiment.exists(
                spec, backend=backend, collection=collection,
                project=project, rerun="stale", run_key=run.run_key,
            )
        if skip:
            logger.warning("run already exists for %s, skipping: %s", name, spec)
            return

    with run:
        resolver = Resolver(spec, factories)
        runnable = resolver.build(entry)
        run.set_graph(resolver.graph)
        observe.checkpoint("init_resolve")

        profiler_cm = Profiler() if profile else nullcontext()
        with profiler_cm as profiler:
            runnable.kernel()

        observe.checkpoint("run_completed")

        if profile and profiler is not None:
            run.add_profile(profiler)

        observe.checkpoint("build_profile")


def run_sweep(name: str, sweep: Sweep) -> None:
    """Run every point in ``bm``'s grid, one :class:`Experiment` each.

    A single point raising (a non-converging solver, a bad input file, ...)
    is logged as a FAILED run and does not abort the rest of a sweep that
    may be running unattended for hours on an HPC cluster. Call
    :func:`execute` directly instead of this function if you want a single
    point's exception to propagate (e.g. while developing a new
    ``Runnable``). A frozen project is the one exception: it aborts the
    sweep instead of failing every point in turn.
    """
    for case in sweep.cases:
        for params in expand_grid(sweep.grid):
            spec = {**case, **params}
            try:
                execute(
                    name, spec, sweep.entry, sweep.factories,
                    backend=sweep.backend, collection=sweep.collection,
                    project=sweep.project, deps=sweep.deps, inputs=sweep.inputs,
                    rerun=sweep.rerun, snapshot=sweep.snapshot,
                    checkpoint=sweep.checkpoint, profile=sweep.profile,
                    overwrite=sweep.overwrite,
                )
            except FrozenProjectError:
                raise
            except Exception:
                logger.exception("run failed for %s, spec=%s; continuing sweep", name, spec)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger()


def run(benchmarks: dict[str, Sweep]) -> None:
    parser = argparse.ArgumentParser(prog='chembench')
    parser.add_argument('-b', '--bench', type=str, help="Specific benchmark to run")
    parser.add_argument('-a', '--array-id', type=int, help="Run benchmark by index mapping (for Slurm arrays)")
    parser.add_argument('--list', action='store_true', help="List all benchmarks and their integer mappings")
    args = parser.parse_args()

    bench_names = sorted(benchmarks.keys())

    if args.list:
        print(f"Total benchmarks: {len(bench_names)}")
        print("Slurm Array Mapping:")
        for idx, name in enumerate(bench_names):
            print(f"  {idx}: {name}")
        return

    logger = setup_logger()
    logger.info("Starting Benchmarks ...")

    if args.array_id is not None:
        if not (0 <= args.array_id < len(bench_names)):
            logger.error("Slurm Array ID %d is out of bounds (Max: %d)", args.array_id, len(bench_names) - 1)
            sys.exit(1)
        target = bench_names[args.array_id]
        logger.info("Mapped SLURM_ARRAY_TASK_ID %d -> %s", args.array_id, target)
        run_sweep(target, benchmarks[target])
    elif args.bench:
        if args.bench not in benchmarks:
            logger.error("Benchmark '%s' not found.", args.bench)
            sys.exit(1)
        run_sweep(args.bench, benchmarks[args.bench])
    else:
        for name, bm in benchmarks.items():
            run_sweep(name, bm)

