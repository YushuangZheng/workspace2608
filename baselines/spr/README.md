# SPR

The official SPRVLA release is fixed at commit
`d57e4b81ebdcacea574b68be29d61ba04cdc7051`. Tasks 0-7 of the fixed
LIBERO-Long suite completed 50 episodes each. Independent validation found 50
official outcomes, annotations, raw videos, and annotated videos for every
completed task, with matching success labels.

| Task | Successes | Episodes | Rate |
|---:|---:|---:|---:|
| 0 | 41 | 50 | 82.0% |
| 1 | 47 | 50 | 94.0% |
| 2 | 38 | 50 | 76.0% |
| 3 | 44 | 50 | 88.0% |
| 4 | 33 | 50 | 66.0% |
| 5 | 42 | 50 | 84.0% |
| 6 | 38 | 50 | 76.0% |
| 7 | 40 | 50 | 80.0% |
| **Available aggregate** | **323** | **400** | **80.75%** |

Task 8 failed during vLLM CUDA-graph warm-up before episode 0 with a CUDA OOM
while GPU 1 had only 14.31 MiB free; it produced no result or video. Task 9 was
never started. Execution was deliberately paused on 2026-08-23, so there is no
verified ten-task aggregate.

Full paper training is not attempted: it used 32 A100-80GB GPUs and the public
repository does not release the complete training recipe or datasets. The
released evaluator also omits one paper-described trajectory-stagnation test,
so results are explicitly labelled `released-code evaluator reproduction`.
The paper publishes 82.8% only for the aggregate over all ten tasks and no
task-level comparator, so the local task-0 value of 82.0% is not a reproduction
of, or a direct comparison with, that paper number.

The isolated environment is `dynamac-spr` (Python 3.11, vLLM 0.8.5). Four
RTX 4090 GPUs match the paper's published inference hardware class.

Completed logs are under ignored
`results/released_code_libero10_task{0..7}/`; task 8's diagnostic OOM log is
`results/released_code_libero10_task8/20260821_194332.log`. Earlier partial
task-0 and task-1 attempts remain diagnostic-only and are excluded from every
metric. The paper's 82.8% ten-task aggregate cannot be compared directly with
the incomplete local eight-task aggregate.
