# M4 failure-supervised monitor

This server-B method directory contains the causal classifier, training-only
views over A-frozen manifests, training loop, checkpoint I/O, and the runtime
adapter. It deliberately does not define task, fault, audit, recovery, runner,
or formal calibration behavior.

The current encoder accepts an already-frozen one-dimensional feature vector.
Its dimension is configured when the model is constructed; no provisional
field names are embedded in the checkpoint. Once server A publishes the
evaluation interface bundle, an A-owned feature encoder can be injected into
`FailureSupervisedMonitor` without changing the model or training loop.

Formal Main-10 training must wait for A's immutable failure-train bundle.
Fault family, severity, trigger time, audit metadata, future samples, normal
calibration episodes, and sealed-test data are not classifier inputs. Server A
alone generates the final threshold and persistence artifacts.

Synthetic tests cover prefix causality, right-padding masks, deterministic
budget views, monitor state reset, training, and checkpoint round trips.
