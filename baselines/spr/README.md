# SPR

The official SPRVLA release is fixed at commit
`d57e4b81ebdcacea574b68be29d61ba04cdc7051`. The first quantitative run uses
the official `SPRVLA/libero_10` checkpoint on one LIBERO-Long task, then all ten
tasks (50 episodes each) if the smoke run is valid.

Full paper training is not attempted: it used 32 A100-80GB GPUs and the public
repository does not release the complete training recipe or datasets. The
released evaluator also omits one paper-described trajectory-stagnation test,
so results are explicitly labelled `released-code evaluator reproduction`.

The isolated environment is `dynamac-spr` (Python 3.11, vLLM 0.8.5). Four
RTX 4090 GPUs match the paper's published inference hardware class.
