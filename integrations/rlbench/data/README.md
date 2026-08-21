# RLBench data

Only the current training and evaluation datasets belong below this directory:

```text
data/
  training/
    manifest.json
    main/<task>/all_variations/episodes/episode{0..4}/
    coordination/bimanual_handover_item/all_variations/episodes/episode{0..4}/
  evaluation/
    manifest.json
    spec.json
    environment/<task>_a_b_n200.json
    coordination/<task>_a_only_n200.json
```

Use `data/training/main` as the default training root. The separate
`data/training/coordination` root contains the dynamic HandOver cohort because
it has the same policy-task alias as the main HandOver cohort.

There are exactly five current demonstrations for each of the eight main tasks,
plus five Coordination demonstrations (45 total). `manifest.json` binds every
retained episode file and its SHA-256. StoreBottle and SweepDust use their V4
replacement cohorts; their superseded Table-II cohorts are not retained.

Training consumes `low_dim_obs.pkl`; the small variation files and collection
manifests are retained for reproducibility. Treat pickle files as trusted
project data and verify data and asset licenses before redistribution.

`data/evaluation` is the only materialized evaluation set. It contains the
latest sealed 200-episode inputs for all eight tasks and the Coordination
initialization batch. OpenMicrowave replacement provenance is embedded in its
current plan envelope; there is no second runnable or external base plan.
