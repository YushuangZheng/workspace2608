from __future__ import annotations

import pytest

from evaluations.development.phase6_formal_evaluation.resources import build_lane_specs


def test_48_workers_reuse_each_gpu_six_times_without_cpu_overlap() -> None:
    specs = build_lane_specs(tuple(range(8)), 48, online_cpus=range(128))

    assert len(specs) == 48
    assert {gpu: sum(spec.gpu == gpu for spec in specs) for gpu in range(8)} == {
        gpu: 6 for gpu in range(8)
    }
    logical_sets = [set(spec.logical_cpus) for spec in specs]
    assert all(logical_sets)
    assert sum(map(len, logical_sets)) == len(set().union(*logical_sets))
    assert all(
        all((cpu < 64) == (cpu in spec.physical_cores) for cpu in spec.logical_cpus)
        for spec in specs
    )


def test_workers_remain_numa_local_to_assigned_gpu() -> None:
    specs = build_lane_specs(tuple(range(8)), 36, online_cpus=range(128))

    for spec in specs:
        if spec.numa_node == 0:
            assert all(0 <= core < 32 for core in spec.physical_cores)
        else:
            assert all(32 <= core < 64 for core in spec.physical_cores)


@pytest.mark.parametrize("workers", [0, 65])
def test_worker_count_requires_one_physical_core_per_lane(workers: int) -> None:
    with pytest.raises(ValueError):
        build_lane_specs(tuple(range(8)), workers, online_cpus=range(128))
