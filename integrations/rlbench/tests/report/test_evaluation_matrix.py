from integrations.rlbench.rlbench_dynamac.report.matrix import PAPER_CELLS


EXPECTED_V4_MATRIX = {
    "stack_wine/static": ("I", 1.00),
    "stack_wine/smooth": ("I", 1.00),
    "stack_wine/teleport": ("I", 1.00),
    "place_cups/static": ("I", 0.99),
    "place_cups/smooth": ("I", 0.97),
    "place_cups/teleport": ("I", 0.99),
    "open_microwave/static": ("I", 0.99),
    "open_microwave/smooth": ("I", 0.99),
    "open_microwave/teleport": ("I", 0.97),
    "wipe_desk/static": ("I", 1.00),
    "wipe_desk/smooth": ("I", 0.66),
    "wipe_desk/teleport": ("I", 0.69),
    "bimanual_put_bottle_in_fridge/static": ("II", 0.82),
    "bimanual_handover_item/static": ("II", 0.97),
    "bimanual_sweep_to_dustpan/static": ("II", 1.00),
    "bimanual_lift_tray/static": ("II", 1.00),
    "bimanual_handover_item_dynamic/coordination_hand_left": ("III", 0.97),
    "bimanual_handover_item_dynamic/coordination_hand_right": ("III", 0.97),
    "bimanual_put_bottle_in_fridge/teleport": ("III", 0.82),
    "bimanual_handover_item/teleport": ("III", 0.97),
    "bimanual_sweep_to_dustpan/teleport": ("III", 1.00),
    "bimanual_lift_tray/teleport": ("III", 1.00),
}


def test_v4_matrix_freezes_exact_cells_tables_and_paper_targets() -> None:
    actual = {
        f"{cell.local_task}/{cell.local_scenarios[0]}": (
            cell.table,
            cell.paper_rate,
        )
        for cell in PAPER_CELLS
    }

    assert len(PAPER_CELLS) == 22
    assert len(actual) == 22
    assert actual == EXPECTED_V4_MATRIX
