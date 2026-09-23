# RLBench Data Layout

The RLBench integration expects demonstrations and evaluation inputs to remain
separate:

```text
data/
├── training/
│   ├── main/<task>/all_variations/episodes/episode{0..4}/
│   └── coordination/<task>/all_variations/episodes/episode{0..4}/
└── evaluation/
    ├── manifest.json
    ├── spec.json
    ├── environment/
    └── coordination/
```

Training episodes contain the low-dimensional observations used to fit the
motion policy and TSF task model. Evaluation inputs contain only sealed episode
initialization and assignment data; evaluation outcomes must never be used as
training input.

Every materialized dataset should provide a manifest with file digests and a
split identifier. Data loaders reject evaluation and result directories when a
training root is requested. Treat pickle files as trusted project data and
verify the applicable dataset, simulator, and asset licenses before including
them in a distribution.

The source package and unit tests do not require the full RLBench dataset. A
minimal runnable package may instead provide one documented task fixture and a
corresponding checkpoint.
