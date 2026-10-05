# tracking

A lightweight, backend-agnostic experiment tracking and grid-sweep driver for numerical experiments, built on top of [Sacred](https://sacred.readthedocs.io).

## Description

`tracking` solves one recurring problem: you have a library that does some numerical computation (a solver, a simulation, a model fit), and you want to run it many times over a cross product of configuration dimensions — algorithm variants, parameters, input datasets — while recording not just the final result but internal metrics, per-phase timings, and whatever the computation chooses to log from deep inside, and you want the results to end up in MongoDB, a local file, or both, interchangeably.

It is built on:

* [Sacred](https://sacred.readthedocs.io) for run documents, metrics and artifact storage
* [PyMongo](https://pymongo.readthedocs.io) for the MongoDB backend
* [pandas](https://pandas.pydata.org) for reading results back out

The design has one rule: **your library never imports `tracking`'s backend.** It only ever talks to `tracking.observe`, a tiny no-op-by-default interface. Whether a run is being tracked at all, and where it ends up, is entirely up to whoever is driving the sweep.

## Installation

The project is not yet published to PyPI, so install it directly from Git.

### Installing into a virtual environment

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/nikplx/tracking.git
```

### Installing into an existing uv project

```shell
uv add git+https://github.com/nikplx/tracking.git
```

## Developer Instructions

```shell
git clone git@github.com:nikplx/tracking.git
cd tracking
uv sync
```

## Basic Usage

### The core idea: `observe`

Your library code never knows whether anything is listening. It just calls into `tracking.observe`, which is a no-op until something activates a run:

```python
from tracking import observe

def solve(matrix, tolerance):
    with observe.measure("factorize"):
        factors = factorize(matrix)

    for i, residual in enumerate(iterate(factors, tolerance)):
        observe.log_metric("residual", residual)

    observe.checkpoint("converged")
    observe.log_metric("iterations", i)
    return factors
```

Call `solve(...)` on its own and every one of these calls is a cheap no-op. Nothing imports Sacred, MongoDB, or anything else — `solve` has no idea `tracking` exists beyond this one import.

### Running one tracked calculation: `Experiment`

To actually record something, wrap a calculation in an `Experiment`. It's a context manager that becomes the active target for every `observe.*` call made anywhere underneath it — including deep inside library code that has no reference to the `Experiment` object itself:

```python
from tracking import Experiment

params = {"matrix": "well_conditioned_512", "tolerance": 1e-8}

with Experiment(name="solver_sweep", params=params, backend="file") as run:
    factors = solve(load_matrix(params["matrix"]), params["tolerance"])
    run.log_metric("rank", factors.rank)
```

Inside the `with` block, `observe.log_metric`/`observe.checkpoint`/`observe.measure` calls made by `solve` land on this exact run, right alongside `run.log_metric("rank", ...)` called directly. Metrics logged as plain scalars (`bool`/`int`/`float`) are additionally recorded as a time series (via Sacred's `log_scalar`), so you get both "the last value" and "the value over the run" for free.

`run.checkpoint(name)` records a named phase boundary with the time elapsed since the previous checkpoint — useful for seeing where time actually goes inside a long calculation, not just the total wall time:

```python
    run.checkpoint("grid_built")
    ... # the expensive part
    run.checkpoint("solved")
```

### Composing a run out of sub-computations: `observe.isolated()`

Some runs aren't a single calculation — a benchmark case might average over several inputs (a reaction over several molecules, a trajectory over several frames, a batch over several samples), and each one logs its own metrics through the exact same `observe` calls. `observe.isolated()` captures everything logged inside the block into its own dict, without touching the active run or leaking keys between sub-computations:

```python
results = []
for sample in batch:
    with observe.isolated() as captured:
        output = solve(sample.matrix, tolerance)
    results.append(captured)  # {"residual": ..., "iterations": ..., ...}

run.log_metric("mean_iterations", mean(r["iterations"] for r in results))
```

### Describing what varies: `Runnable`, factories, and `Resolver`

A single calculation is a `Runnable` — anything with a no-argument `kernel()` method:

```python
from tracking import Runnable

class IterativeSolver(Runnable):
    def __init__(self, matrix, tolerance, preconditioner=None):
        self.matrix = matrix
        self.tolerance = tolerance
        self.preconditioner = preconditioner

    def kernel(self) -> None:
        solve(self.matrix, self.tolerance, self.preconditioner)
```

You rarely construct one by hand. Instead you register a `factories` dict mapping a name to whatever builds that piece, and a `Resolver` builds the one you ask for, resolving its dependencies recursively by inspecting constructor signatures — the same way `Experiment`'s own params work, generalized to a full dependency graph:

```python
from tracking import Resolver

factories = {
    "matrix": load_matrix,          # def load_matrix(matrix_path) -> Matrix
    "preconditioner": build_jacobi, # def build_jacobi(matrix) -> Preconditioner
    "run": IterativeSolver,
}

spec = {"matrix_path": "well_conditioned_512.npz", "tolerance": 1e-8}
runnable = Resolver(spec, factories).build("run")
runnable.kernel()
```

`Resolver` asks for `"run"` → needs `matrix`, `tolerance`, `preconditioner` → `tolerance` is already in `spec`, `matrix` and `preconditioner` are built from their own factories (each resolving *their* dependencies the same way), and the whole graph is built in the right order regardless of which key you ask for first. Every value a literal in `spec` always wins over a factory with the same name, and each key is only built once per `Resolver` (so `matrix` is built once even though both `IterativeSolver` and `build_jacobi` depend on it).

### Describing a sweep: `Sweep` and the grid

A `Sweep` ("benchmark") is one named family of runs: a list of fixed `cases` (e.g. one per input dataset) crossed with a `grid` of parameters that gets fully expanded — add a value to any list and every existing combination now also runs with it:

```python
from tracking import Sweep, run_sweep

bm = Sweep(
    cases=[
        {"matrix_path": "well_conditioned_512.npz"},
        {"matrix_path": "ill_conditioned_512.npz"},
    ],
    grid={
        "tolerance": [1e-6, 1e-8, 1e-10],
        "preconditioner_kind": ["none", "jacobi"],
    },
    factories=factories,
    entry="run",
    backend="mongodb",
    collection="solver_sweep_v1",
)

run_sweep("solver_sweep", bm)
```

This runs `2 cases × 3 tolerances × 2 preconditioners = 12` points, one `Experiment` each. A run that already completed with the exact same spec (checked against MongoDB) is skipped, so re-running the same sweep after adding one more `tolerance` value only runs the new points. A point that raises (a solver that fails to converge, a bad input file) is recorded as a `FAILED` run and the sweep keeps going — call `execute(...)` directly instead of `run_sweep` if you want a failure to stop you immediately, e.g. while developing a new `Runnable`.

### Running a sweep from the command line

`run(benchmarks)` turns a `dict[str, BM]` into a small CLI, including `--array-id` for SLURM job arrays (one array index per named `Sweep`, not per grid point — each `Sweep`'s own grid still runs sequentially within that task):

```python
# sweep.py
from tracking import run

BENCHMARKS = {"solver_sweep": bm, "other_sweep": other_bm}

if __name__ == "__main__":
    run(BENCHMARKS)
```

```shell
python sweep.py --list                 # show the name <-> array-id mapping
python sweep.py --bench solver_sweep   # run one named sweep
python sweep.py --array-id 0           # run whichever sweep SLURM mapped to this index
python sweep.py                        # run everything
```

### Backends

`backend="mongodb"` (the default) writes each run as a Sacred document via `QueuedMongoObserver`, queuing writes so a flaky HPC network doesn't lose data. It needs a connection string in the `MONGODB_URL` or `MONGO_URL` environment variable, and defaults to the `tracking` database — override with `TRACKING_MONGO_DB`. If no URL is set, it logs a warning and falls back to the file backend rather than failing the run.

`backend="file"` writes a full Sacred run directory under `./sacred_runs/<id>/` (config, metrics, captured output) and appends one flat JSON row per run to `./experiments.jsonl` in the current directory, for quick loading without walking the Sacred directory tree.

Reading a MongoDB collection back into a flat `pandas.DataFrame` (one `dimensions_*`/`metrics_*` column per config/info key):

```python
from tracking.experiment import get_runs_collection
from tracking import load_runs_df

df = load_runs_df(get_runs_collection("solver_sweep_v1"))
df.groupby("dimensions_tolerance")["metrics_iterations"].mean()
```

## Notes

* `Experiment` drives Sacred's run lifecycle itself rather than going through `ex.run()`, so it can create one run per grid point without a single fixed `@ex.main` and without Sacred's background heartbeat thread (it flushes at each checkpoint instead). This relies on a handful of private Sacred methods, which is why `sacred` is pinned to an exact version (`==0.8.7`) rather than a range — bumping it needs re-checking that `Experiment.__enter__`/`__exit__` still match Sacred's internals.
* The Sacred `Experiment` object (`tracking.experiment.ex`) is a single module-level instance shared by every `tracking.Experiment` in a process. Run sweeps as separate processes (e.g. one per SLURM array task, as `run()` does) rather than from multiple threads in the same process.
