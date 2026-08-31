"""Hardware lane allocation shared by formal execution and stress probes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence


# Local host topology: GPU 0..2 are attached to NUMA node 0 and GPU 3..7 to
# NUMA node 1.  Logical CPU i+64 is the SMT sibling of physical CPU i.
GPU_NUMA_NODE = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
PHYSICAL_CORES_BY_NUMA = {
    0: tuple(range(0, 32)),
    1: tuple(range(32, 64)),
}
SMT_OFFSET = 64
MAX_WORKERS = sum(len(value) for value in PHYSICAL_CORES_BY_NUMA.values())


@dataclass(frozen=True)
class LaneSpec:
    """One isolated simulator worker and its reusable hardware identity."""

    lane: int
    gpu: int
    numa_node: int
    physical_cores: tuple[int, ...]
    logical_cpus: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "gpu": self.gpu,
            "numa_node": self.numa_node,
            "physical_cores": list(self.physical_cores),
            "logical_cpus": list(self.logical_cpus),
        }


def _partition(values: Sequence[int], groups: int) -> tuple[tuple[int, ...], ...]:
    if groups < 1 or groups > len(values):
        raise ValueError("each worker requires at least one physical CPU core")
    quotient, remainder = divmod(len(values), groups)
    result = []
    offset = 0
    for index in range(groups):
        width = quotient + (1 if index < remainder else 0)
        result.append(tuple(values[offset : offset + width]))
        offset += width
    return tuple(result)


def build_lane_specs(
    gpus: Sequence[int],
    workers: int,
    *,
    online_cpus: Iterable[int] | None = None,
) -> tuple[LaneSpec, ...]:
    """Allocate non-overlapping NUMA-local physical cores to worker lanes.

    GPU identities are reused round-robin.  CPU cores are never shared between
    lanes; every allocated physical core carries its SMT sibling when online.
    This makes worker count a throughput setting without allowing hidden CPU
    oversubscription to change simulator timing.
    """

    gpu_ids = tuple(int(value) for value in gpus)
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU identities must be a non-empty distinct sequence")
    unknown = sorted(set(gpu_ids).difference(GPU_NUMA_NODE))
    if unknown:
        raise ValueError(f"GPU NUMA mapping is unavailable: {unknown}")
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must lie in [1,{MAX_WORKERS}]")

    assigned_gpus = tuple(gpu_ids[index % len(gpu_ids)] for index in range(workers))
    lane_ids_by_node = {
        node: tuple(
            index
            for index, gpu in enumerate(assigned_gpus)
            if GPU_NUMA_NODE[gpu] == node
        )
        for node in PHYSICAL_CORES_BY_NUMA
    }
    online = set(os.sched_getaffinity(0) if online_cpus is None else online_cpus)
    cores_by_lane: dict[int, tuple[int, ...]] = {}
    for node, lane_ids in lane_ids_by_node.items():
        if not lane_ids:
            continue
        available_physical = tuple(
            core
            for core in PHYSICAL_CORES_BY_NUMA[node]
            if core in online or core + SMT_OFFSET in online
        )
        chunks = _partition(available_physical, len(lane_ids))
        cores_by_lane.update(zip(lane_ids, chunks))

    result = []
    used_logical: set[int] = set()
    for lane, gpu in enumerate(assigned_gpus):
        physical = cores_by_lane[lane]
        logical = tuple(
            cpu
            for core in physical
            for cpu in (core, core + SMT_OFFSET)
            if cpu in online
        )
        if not logical:
            raise RuntimeError(f"lane {lane} has no online logical CPU")
        overlap = used_logical.intersection(logical)
        if overlap:
            raise RuntimeError(f"lane CPU affinity overlaps: {sorted(overlap)}")
        used_logical.update(logical)
        result.append(
            LaneSpec(
                lane=lane,
                gpu=gpu,
                numa_node=GPU_NUMA_NODE[gpu],
                physical_cores=physical,
                logical_cpus=logical,
            )
        )
    return tuple(result)


__all__ = [
    "GPU_NUMA_NODE",
    "LaneSpec",
    "MAX_WORKERS",
    "build_lane_specs",
]
