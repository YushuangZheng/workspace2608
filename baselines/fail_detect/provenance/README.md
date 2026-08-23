# Quantitative provenance

The bounded pipeline writes
`external_dp_logpzo_v1.json` here after the gate and updates it after the final
50+50 run. This small, reviewable file records source commits, official URLs,
runtime SHA-256 values, generated feature/detector identities, and the exact
bounded protocol. Large source artifacts, checkpoints, rollouts, and full
results remain ignored.

The generated JSON is intentionally not pre-populated: it must only be reviewed
and committed after hashes have been calculated from the actual server files.
