# RLBench Demonstrations

Set `DYNAMAC_DATA_ROOT` to a directory containing one standard RLBench episode tree per task:

```text
$DYNAMAC_DATA_ROOT/
  <task>/
    all_variations/
      episodes/
        episode0/
          low_dim_obs.pkl
        episode1/
          low_dim_obs.pkl
        ...
```

Training uses `low_dim_obs.pkl` only. Camera observations may remain beside the low-dimensional data for inspection but are not policy inputs. Left- and right-arm observations must be paired by episode and time step.

The default Table II experiment uses five demonstrations for each of these tasks:

- `bimanual_put_bottle_in_fridge`
- `bimanual_handover_item`
- `bimanual_sweep_to_dustpan`
- `bimanual_lift_tray`

RLBench uses sample-then-verify generation, so regenerating a dataset requires seeding every random source in the collection pipeline. Treat pickle files as trusted project data and verify data and asset licenses before redistribution.
