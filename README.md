# tracking

A lightweight experiment tracking and grid-sweep driver for numerical experiments, backed by a compact MongoDB document per run.

## Description

`tracking` solves one recurring problem: you have a library that does some numerical computation (a solver, a simulation, a model fit), and you want to run it many times over a cross product of configuration dimensions — algorithm variants, parameters, input datasets — while recording not just the final result but internal metrics, per-phase timings, provenance (code, environment, inputs), and any large artifacts (matrices, profiler reports) it chooses to save from deep inside.

It is built on:

* [PyMongo](https://pymongo.readthedocs.io) for the compact MongoDB backend
* a content-addressed object store — a local directory, or an S3-compatible bucket — for artifacts, input files and source-tree snapshots
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

Call `solve(...)` on its own and every one of these calls is a cheap no-op. Nothing imports MongoDB or anything else — `solve` has no idea `tracking` exists beyond this one import.

### Running one tracked calculation: `Experiment`

To actually record something, wrap a calculation in an `Experiment`. It's a context manager that becomes the active target for every `observe.*` call made anywhere underneath it — including deep inside library code that has no reference to the `Experiment` object itself:

```python
from tracking import Experiment

params = {"matrix": "well_conditioned_512", "tolerance": 1e-8}

with Experiment(name="solver_sweep", params=params) as run:
    factors = solve(load_matrix(params["matrix"]), params["tolerance"])
    run.log_metric("rank", factors.rank)
```

Inside the `with` block, `observe.log_metric`/`observe.checkpoint`/`observe.measure` calls made by `solve` land on this exact run, right alongside `run.log_metric("rank", ...)` called directly. Each call stores the latest value per metric; use `run.append_metric` / `observe.append_metric` when you explicitly need a list of values.

`run.checkpoint(name)` records a named phase boundary with the time elapsed since the previous checkpoint — useful for seeing where time actually goes inside a long calculation, not just the total wall time:

```python
    run.checkpoint("grid_built")
    ... # the expensive part
    run.checkpoint("solved")
```

### Storing large results: artifacts

Matrices, Wannier90 files, profiler reports — anything too big or too binary for a Mongo document — go through `add_artifact`/`add_array` instead of `log_metric`. The bytes are written to the project's content-addressed store (see [Backends and storage](#backends-and-storage)); only a small descriptor (`name`, `filename`, `sha256`, `size`, `object`, ...) is embedded in the run document. `object` is relative to the store root (for example `objects/ab/<sha>.npz`), so a copied or shipped store needs no path rewriting:

```python
run.add_array("density_matrix", {"dm": factors.density_matrix}, axes="ao,ao")
run.add_artifact("wannier90.win", wannier_input_path, media_type="text/plain")
```

`observe.add_artifact`/`observe.add_array` work the same way from inside `solve`, dispatching to whichever run is active. Read one back later with `Project.load_artifact(descriptor)` (raw bytes, or a materialized file when given `dest=`), or list every artifact across a collection with `Project.artifacts(collection)`.

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

`Resolver` asks for `"run"` → needs `matrix`, `tolerance`, `preconditioner` → `tolerance` is already in `spec`, `matrix` and `preconditioner` are built from their own factories (each resolving *their* dependencies the same way), and the whole graph is built in the right order regardless of which key you ask for first. Every literal value in `spec` always wins over a factory with the same name, and each key is only built once per `Resolver` (so `matrix` is built once even though both `IterativeSolver` and `build_jacobi` depend on it).

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
    backend="mongo_compact",
    collection="solver_sweep_v1",
)

run_sweep("solver_sweep", bm)
```

This runs `2 cases × 3 tolerances × 2 preconditioners = 12` points, one `Experiment` each. A run that already completed with the exact same spec (checked against the configured backend via its spec hash) is skipped, so re-running the same sweep after adding one more `tolerance` value only runs the new points. A point that raises (a solver that fails to converge, a bad input file) is recorded as a `FAILED` run and the sweep keeps going — call `execute(...)` directly instead of `run_sweep` if you want a failure to stop you immediately, e.g. while developing a new `Runnable`.

Other `Sweep` fields: `checkpoint=True` flushes at every `observe.checkpoint` (useful for long HPC runs), `profile=True` attaches a [pyinstrument](https://github.com/joerick/pyinstrument) HTML report as an artifact, and `overwrite=True` skips the completed-run check and reruns every point.

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

### Backends and storage

`backend="mongo_compact"` (the default) writes **one self-contained document per run** to MongoDB, keyed by `spec_hash`. `backend="file"` writes the same document as a local JSON file instead — no MongoDB needed — under `TRACKING_FILE_ROOT` (default `DATA_DIR/runs`)::

    <file_root>/<collection>/<run_key>.json

The file backend is atomic and checkpoint-aware just like Mongo (a `.json` file is replaced on entry, at every `observe.checkpoint`, and on exit), and `Project.load`/`status`/`artifacts` work unchanged when the `Project` is created with `backend="file"`:

```python
from tracking import Experiment, Project, load_file_data

project = Project("paper", backend="file", file_root="runs")
with Experiment("solver", {"tol": 1e-8}, backend="file", project=project) as run:
    run.log_metric("iterations", 12)

df = load_file_data(project, collection="runs")   # or project.load("runs")
```

The document holds `meta`, `dimensions`, `metrics`, `checkpoints`, `artifacts`, `provenance`, `status` (`RUNNING`/`COMPLETED`/`FAILED`) and `updated`. The Mongo backend is written on entry, at every `observe.checkpoint`, and on exit, with retrying writes so a flaky HPC network does not lose progress. Connection settings:

* `MONGODB_URL` or `MONGO_URL` — connection string (required for the Mongo backend; a clear error is raised if neither is set)
* `TRACKING_MONGO_DB` — database name (default `tracking`), or `TRACKING_PROJECT` to select a `Project` by name
* `TRACKING_FILE_ROOT` — root directory for `backend="file"` (default `DATA_DIR/runs`)
* `collection=` on `Experiment`/`Sweep` — collection name (default `runs`)

Older collections written by the historical Sacred-document and `backend="mongodb"` schemas are still readable through `normalize_doc` / `load_mongo_data`.

Artifacts, input files and source-tree snapshots go through a separate, content-addressed object store, configured via `TRACKING_STORE`:

* a plain path (or `DATA_DIR/tracking_store` by default) — backed by `tracking.store.LocalStore`, good enough on a shared HPC filesystem;
* an `s3://bucket/prefix` URI — backed by `tracking.store_s3.S3Store` against any S3-compatible endpoint (AWS, MinIO, Ceph). Requires the `tracking[s3]` extra (`pip install tracking[s3]`), which pulls in `boto3`.

Both implement the same protocol (`tracking.store.Store`), so nothing else in the package — or your own code reading `Project.store`/`Project.load_artifact` — needs to know which one is in use. Local objects keep the source extension (`objects/ab/<sha>.npz`, `objects/cd/<sha>.html`, ...), while the hash remains the identity, so the store is browsable and materialized artifacts open with the right application.

### Reading results back

`load_mongo_data`/`normalize_doc` flatten a collection into a `pandas.DataFrame` with one `dimensions_*` / `metrics_*` / `provenance_*` column per key:

```python
from tracking import load_mongo_data
from tracking.experiment import get_runs_collection

mongo_df = load_mongo_data(
    get_runs_collection("solver_sweep_v1"),
    {"status": "COMPLETED"},  # pass {} to read every status
)
mongo_df.groupby("dimensions_tolerance")["metrics_iterations"].mean()
```

`get_runs_collection(name)` uses the same `MONGODB_URL`/`MONGO_URL` and `TRACKING_MONGO_DB` settings as the compact backend. `load_mongo_data(collection, query)` needs an explicit query. For local files, `load_file_data(project, collection="runs", query=...)` returns the same normalized DataFrame shape. `Project.status(collection)` additionally computes a `validity` column (`CURRENT`/`STALE_CODE`/`STALE_ENV`/`STALE_INPUT`/`INVALIDATED`/`UNKNOWN`) by re-checking each run's recorded provenance against the current code and inputs. `load_json_data(path)` remains available as a generic JSONL loader for externally produced files, skipping a half-written final line from a live writer.

Artifacts are just as easy to get at. `Project.load_artifacts(collection, query=...)` returns one row per artifact with its descriptor fields plus a `filename` and a ready-to-open local `path`; pass `dest="export/"` to materialize copies with their proper endings. No hashes to wrangle:

```python
artifacts = project.load_artifacts("solver_sweep_v1")
artifacts[["name", "filename", "path"]]

# or from tracking import load_artifacts
array_path = project.load_artifacts("solver_sweep_v1", dest="artifacts").at[0, "path"]
```


### Comparing runs against a baseline

`add_reference_diffs` joins each run to the matching reference row (same `keys`) and adds `<metric>_vs_ref`, `<metric>_vs_ref_pct`, and `<metric>_vs_ref_abs` columns. By default it auto-discovers `metrics_*` columns and returns only the non-reference rows; pass `return_reference=True` to keep both, or `metrics=[...]` to choose columns explicitly:

```python
from tracking import add_reference_diffs

diffed = add_reference_diffs(
    mongo_df,
    group="dimensions_preconditioner_kind",
    reference="none",
    keys=["dimensions_matrix_path", "dimensions_tolerance"],
)
```

## Notes

* Every run's `provenance` records `env` (Python + package versions) and `system` (hostname, OS/kernel/arch, CPU, memory, and — when present — SLURM job/array/node fields and container runtime), captured best-effort so one failing probe never breaks a run. Neither is part of `run_key`/`spec_hash`: two runs on different hosts with identical code and inputs still dedupe against each other. Set `TRACKING_SYSTEM_INFO=0` to skip system capture entirely.
