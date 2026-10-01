from abc import ABC, abstractmethod

from tracking.configurable import Configurable

class Runnable(ABC, Configurable):
    @abstractmethod
    def kernel(self) -> None:
        """Run the calculation and log whatever it wants via ``tracking.observe``.

        No return value is part of the contract: callers that only drive
        the run (the benchmark dispatcher) never inspect it, only that it
        completed without raising. A ``Runnable`` that composes sub-runs
        (e.g. a batch averaged over several inputs) may still use a return
        value for its *own* bookkeeping -- that's between it and whatever
        calls it directly, not something the framework relies on.
        """
        pass
