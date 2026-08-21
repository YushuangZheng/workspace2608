"""Authoritative 22-cell V4 paper comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperCell:
    """One local V4 cell and its corresponding paper success rate."""

    table: str
    condition: str
    task: str
    paper_rate: float
    local_task: str
    local_scenarios: tuple[str, ...]
    result_family: str
    note: str = ""


def _build_cells() -> tuple[PaperCell, ...]:
    cells: list[PaperCell] = []
    for task, label, static, smooth, teleport in (
        ("stack_wine", "StackWine", 1.00, 1.00, 1.00),
        ("place_cups", "PlaceCups", 0.99, 0.97, 0.99),
        ("open_microwave", "OpenMicrowave", 0.99, 0.99, 0.97),
        ("wipe_desk", "WipeDesk", 1.00, 0.66, 0.69),
    ):
        for condition, scenario, paper_rate in (
            ("Static", "static", static),
            ("Smooth dynamics", "smooth", smooth),
            ("Teleportation", "teleport", teleport),
        ):
            cells.append(
                PaperCell(
                    table="I",
                    condition=condition,
                    task=label,
                    paper_rate=paper_rate,
                    local_task=task,
                    local_scenarios=(scenario,),
                    result_family="table_i",
                    note=(
                        "Independent five-demonstration cohort."
                        if scenario == "static"
                        else "Local preserve-instance task-root motion schedule; "
                        "paper defaults unpublished."
                    ),
                )
            )

    bimanual = (
        ("bimanual_put_bottle_in_fridge", "StoreBottle", 0.82),
        ("bimanual_handover_item", "HandOver", 0.97),
        ("bimanual_sweep_to_dustpan", "SweepDust", 1.00),
        ("bimanual_lift_tray", "LiftTray", 1.00),
    )
    for task, label, paper_rate in bimanual:
        cells.append(
            PaperCell(
                table="II",
                condition="Static",
                task=label,
                paper_rate=paper_rate,
                local_task=task,
                local_scenarios=("static",),
                result_family="bimanual_static",
                note="Independent five-demonstration cohort.",
            )
        )

    cells.extend(
        (
            PaperCell(
                table="III",
                condition="Coordination",
                task="Hand Left",
                paper_rate=0.97,
                local_task="bimanual_handover_item_dynamic",
                local_scenarios=(
                    "coordination_hand_left",
                    "coordination_left",
                    "hand_left",
                    "perturb_left",
                    "left_arm_perturbed",
                ),
                result_family="bimanual_coordination",
                note="Arm-perturbation magnitude and timing are unpublished.",
            ),
            PaperCell(
                table="III",
                condition="Coordination",
                task="Hand Right",
                paper_rate=0.97,
                local_task="bimanual_handover_item_dynamic",
                local_scenarios=(
                    "coordination_hand_right",
                    "coordination_right",
                    "hand_right",
                    "perturb_right",
                    "right_arm_perturbed",
                ),
                result_family="bimanual_coordination",
                note="Arm-perturbation magnitude and timing are unpublished.",
            ),
        )
    )
    for task, label, paper_rate in bimanual:
        cells.append(
            PaperCell(
                table="III",
                condition="Dynamic environment",
                task=label,
                paper_rate=paper_rate,
                local_task=task,
                local_scenarios=("teleport",),
                result_family="bimanual_dynamic",
                note=(
                    "Local public-RLBench intervention; paper DynaBench "
                    "defaults unpublished."
                ),
            )
        )
    if len(cells) != 22:
        raise RuntimeError("V4 paper matrix must contain exactly 22 cells")
    return tuple(cells)


PAPER_CELLS = _build_cells()

__all__ = ["PAPER_CELLS", "PaperCell"]
