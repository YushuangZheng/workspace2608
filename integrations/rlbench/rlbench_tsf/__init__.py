"""RLBench-only adapters for the environment-neutral TSF policy."""

__all__ = ["TSFObservationAdapter"]


def __getattr__(name):
    """Keep pure simulator adapters importable in the pinned Python 3.8 process."""

    if name == "TSFObservationAdapter":
        from .observation_adapter import TSFObservationAdapter

        return TSFObservationAdapter
    raise AttributeError(name)
