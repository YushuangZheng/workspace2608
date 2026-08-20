# SPR

The official SPRVLA release is fixed at commit
`d57e4b81ebdcacea574b68be29d61ba04cdc7051`. The canonical task-0 rerun
launched at `20260820_070434` completed all 50 fixed LIBERO-Long episodes with
41 successes (82.0%). Its official log is
`results/released_code_libero10_task0/20260820_070434.log`; annotations and the
raw and annotated videos are under
`results/released_code_libero10_task0/rollouts/2026_08_20/`. Independent
validation found exactly 50 official outcomes, 50 annotations, 50 raw videos,
and 50 annotated videos, with matching success labels.

Full paper training is not attempted: it used 32 A100-80GB GPUs and the public
repository does not release the complete training recipe or datasets. The
released evaluator also omits one paper-described trajectory-stagnation test,
so results are explicitly labelled `released-code evaluator reproduction`.
The paper publishes 82.8% only for the aggregate over all ten tasks and no
task-level comparator, so the local task-0 value of 82.0% is not a reproduction
of, or a direct comparison with, that paper number.

The isolated environment is `dynamac-spr` (Python 3.11, vLLM 0.8.5). Four
RTX 4090 GPUs match the paper's published inference hardware class.

An earlier task-0 attempt was externally terminated at 35/50 and is retained
only as diagnostic evidence. A later orchestration interruption occurred after
the canonical 50/50 run had already completed and passed validation, so it does
not affect that result.

To close the requested scope, the later task-1 run was deliberately stopped
after its first completed episode; that episode succeeded. Its partial artifacts
remain under ignored
`results/released_code_libero10_task1_partial_user_scope_stop_20260820_215730/`
for diagnosis only and are excluded from every metric. Tasks 2-9 were not run.
There is therefore no verified ten-task aggregate or comparison chart, and the
paper's 82.8% ten-task aggregate cannot be compared directly with the canonical
task-0 82.0% result.
