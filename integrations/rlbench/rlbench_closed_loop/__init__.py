"""RLBench-only adapters for the environment-neutral closed-loop policy."""

__all__ = ["ClosedLoopObservationAdapter"]


def __getattr__(name):
    """Keep pure simulator adapters importable in the pinned Python 3.8 process."""

    if name == "ClosedLoopObservationAdapter":
        from .observation_adapter import ClosedLoopObservationAdapter

        return ClosedLoopObservationAdapter
    raise AttributeError(name)
