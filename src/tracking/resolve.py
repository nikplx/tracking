"""Generic, recursive dependency resolution -- replaces the old
``hydrate_dependencies`` dict-mutation hack.

The core idea: a benchmark run is built from two separate things.

* ``spec`` -- plain, literal data: file paths, basis-set strings, floats,
  already-built objects a caller wants to hand in directly. Nothing in here
  is ever a class to be instantiated.
* ``factories`` -- a registry mapping a key name to the class or function
  that builds it (``'matrix' -> load_matrix``, ``'run' -> IterativeSolver``, ...).

Requesting a key checks ``spec`` first (a literal always wins), otherwise
calls the matching factory, resolving *its* parameters the same way,
recursively, by inspecting its signature. This generalises
``tracking.configurable.Configurable.from_config`` (which does the same
signature-filtering, but only one level deep) into a real, order-independent
dependency graph -- so it doesn't matter whether ``matrix`` or
``preconditioner`` gets resolved "first"; whichever is asked for first
resolves what it needs on demand, and everything is memoized per
:class:`Resolver` instance.

A factory may also declare ``**kwargs`` (a ``VAR_KEYWORD`` parameter). It
then receives the residual spec: every plain spec entry that is neither
consumed as a named parameter nor provided by a factory. That is the
forwarding mechanism for composing runnables -- e.g. a reaction runner
captures grid knobs it doesn't know by name (``n_laplace``,
``cholesky_threshold``, ...) and merges them into each member's child spec.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

Factory = Callable[..., Any]


class ResolutionError(Exception):
    pass


class Resolver:
    def __init__(self, spec: dict[str, Any], factories: dict[str, Factory]):
        self.spec = spec
        self.factories = factories
        self._built: dict[str, Any] = {}

    def build(self, key: str) -> Any:
        return self._build(key, stack=())

    def _build(self, key: str, stack: tuple[str, ...]) -> Any:
        if key in self._built:
            return self._built[key]

        if key in self.spec:
            value = self.spec[key]
            self._built[key] = value
            return value

        if key not in self.factories:
            raise ResolutionError(f"no spec value or factory registered for {key!r}")

        if key in stack:
            raise ResolutionError(f"dependency cycle: {' -> '.join((*stack, key))}")

        factory = self.factories[key]
        ctor = factory.__init__ if isinstance(factory, type) else factory
        params = inspect.signature(ctor).parameters

        kwargs = {
            name: self._build(name, stack + (key,))
            for name, p in params.items()
            if name != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and (name in self.spec or name in self.factories)
        }

        var_kw = next(
            (name for name, p in params.items()
             if p.kind == inspect.Parameter.VAR_KEYWORD),
            None,
        )
        if var_kw is not None:
            # Residual config capture: forward plain spec entries that are
            # neither consumed as named params nor provided by a factory.
            # Splat them as individual keywords so they land *inside* the
            # callee's **kwargs (assigning under the kwargs-param name would
            # nest the dict one level too deep). This is how a composing
            # Runnable (e.g. a reaction runner that re-enters a child
            # Resolver per member) receives grid knobs it doesn't know by
            # name -- they flow through to the child spec. Factory-named
            # keys are excluded on purpose: they are dependencies the child
            # builds itself (possibly per member), not config to inherit.
            kwargs.update({
                k: self._build(k, stack + (key,))
                for k in self.spec
                if k not in kwargs and k != var_kw and k not in self.factories
            })

        obj = factory(**kwargs)
        self._built[key] = obj
        return obj
