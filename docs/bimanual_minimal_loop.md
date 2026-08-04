# Bimanual DynaMAC minimal loop

The bimanual extension contains exactly the two tasks selected after the single-arm loop passed: Handover and Lift
Tray. Both use two independent Franka arms, a 16-D absolute Cartesian action, lightweight virtual attachment, and
fresh Isaac Lab processes for each collection/evaluation episode. The attachment is geometric rather than a claim
of contact-rich grasp physics.

## Handover

The frozen five-demo dataset is `data/handover_static/v1`. Its 13 states cover left grasp, left transport, right
rendezvous, carrier transfer, right placement, release, and retreat. Verify or reproduce with:

```bash
python scripts/audit_handover_dataset.py --data_dir data/handover_static/v1
python scripts/eval_handover.py --headless --seeds 8208
```

The controlled pilot compares independent arms, a fixed handover point, a fixed-schedule static cross-arm stream,
and Full DynaMAC. Perturbations include left/right offsets, a shifted handover, left/right pauses, and smooth/sudden
single-arm offsets. Full DynaMAC captures the observed right-hand-to-object transform at transfer and solves the
placement target using that live connection geometry.

The final Full DynaMAC development run succeeded in all eight conditions with 5.6--26.9 mm final error. This is a
one-seed engineering result; the independent/fixed/static baselines and all raw trial metrics remain in
`outputs/handover_minimal` when reproduced locally.

| Method | Successful conditions / 8 | Notable failure |
|---|---:|---|
| Independent arms | 7 | shifted handover |
| Fixed handover | 7 | shifted handover |
| Static cross-arm | 7 | sudden right-arm offset |
| Full DynaMAC | 8 | none in this pilot |

## Lift Tray

The frozen dataset is `data/lift_tray_static/v1`. Its nine states cover simultaneous approach, bilateral grasp,
lift, transport, lower, release, and retreat. The tray pose is driven by the midpoint of both grippers while the
connection is active, making arm disagreement directly measurable.

```bash
python scripts/audit_lift_tray_dataset.py --data_dir data/lift_tray_static/v1
python scripts/eval_lift_tray.py --headless --seeds 10208
```

The ablation compares independent arms, an intentionally static shared-object stream, and Full DynaMAC using
captured virtual gripper frames plus the opposite gripper frame during shared transport. Report final placement,
cross-arm width error, total path length, perturbation recovery, and inference time. As with Handover, the current
matrix is a pilot and must be expanded to multiple seeds before paper-level claims.

| Method | Successful conditions / 5 | Static width error | Interpretation |
|---|---:|---:|---|
| Independent arms | 5 | 55.7 mm | placement works, weaker bilateral synchronization |
| Static shared-object | 0 | 32.8 mm | endogenous object feedback makes the path diverge |
| Full DynaMAC | 5 | 29.2 mm | masks the shared-object loop and uses cross-arm/virtual frames |

## Claim boundary

These environments can support claims about reproduced relative-geometry behavior, online reference validity,
zero-shot response to test-time perturbations from static demonstrations, and dynamic cross-arm coordination in the
custom Isaac Lab tasks. They do not establish the paper's reported 35-point gain, 20x sample efficiency, a complete
DynaBench reproduction, or a complete MiDiGaP/Diffusion Policy reproduction.
