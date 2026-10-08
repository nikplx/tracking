# Vendored `sacred`

`tracking/_vendor/sacred/` is a full copy of the [`sacred`](https://github.com/IDSIA/sacred)
package, version **0.8.7** (the last upstream release), which is MIT-licensed —
see `sacred-LICENSE-MIT.txt`.

## Why vendored

Upstream sacred is unmaintained and broke on Python 3.14 / recent setuptools
(`pkgutil.find_loader` was removed; `pkg_resources` is deprecated and slated
for removal). `tracking` drives sacred's run lifecycle directly
(`Experiment._create_run`, observer `started`/`heartbeat`/`completed`/`failed`
events), so it needs a working copy. Vendoring keeps the Mongo document shape
and `sacred_runs/` file layout byte-identical while removing the dead
dependency.

## Adaptations vs. upstream 0.8.7

1. **Package renamed** `sacred` -> `tracking._vendor.sacred` (mechanical rewrite
   of all absolute `import sacred...` / `from sacred...` statements), so the
   vendored copy no longer shadows or conflicts with any PyPI-installed
   `sacred`.
2. **`utils.module_exists`**: `pkgutil.find_loader` (removed in Python 3.12+)
   replaced with `importlib.util.find_spec`.
3. **`observers/mongo.py`**: `pkg_resources.resource_filename("sacred", ...)`
   (deprecated setuptools API) replaced with `importlib.resources.files(...)`
   for locating `data/mime.types`.
4. **`dependencies.py`**: `pkg_resources.working_set` (deprecated setuptools
   API) replaced with `importlib.metadata` (`packages_distributions()` /
   `version()` / `distributions()`).
5. **`utils._is_sacred_frame`**: also matches `tracking._vendor.sacred*` so
   sacred-internal frames keep being filtered from formatted tracebacks.

No behavioral changes beyond the above: the run-document schema, observer
event payloads, and file-observer layout are untouched.

One known side effect of vendoring: because the Sacred code now lives under
`src/tracking/`, Sacred's source discovery records the vendored files as run
sources alongside `tracking`'s own modules (previously Sacred's code lived in
`site-packages` and was excluded as non-local). This only affects the stored
`sources` lists / `_sources` copies, not configs, metrics, or artifacts.

## Updating

If upstream ever releases a fixed version, re-copy the package over this
directory, re-apply the adaptations listed above (each is marked in the code
with a `# vendored:` comment), and re-run the file-backend smoke test
(a single `tracking.Experiment(backend="file")` run plus `normalize_doc`
against a stored Mongo document).
