# A Cycle2 T6 Independent Profile Packaging

Date: 2026-11-02
Branch: `feature/a-mlops`

## Delivery

- `stage_submission.py` now copies `profile_model_dir` when the selected config uses `profile_mode=independent`, rewrites it to the staged relative path, and rejects a blank model path before copying.
- Generated launchers shell-quote the chosen Python executable path.
- `cloud_preflight.py --verify-hashes` checks the complete independent profile checkpoint tree against the pinned v1 digest.
- `Dockerfile.integrated` includes the fixed v1 profile checkpoint from the named `model_assets` context and preflights it with `configs/integrated_profile_v1_off.json`.
- The deployed default `configs/integrated.json` remains `profile_mode=empty`.

## Verification

- Docker/model path and cloud staging/preflight tests: **10/10 passed**.
- The stage test uses small substitute model files, stages the independent profile config, and runs `cloud_preflight --files-only` from the staged directory.
- A shell path containing spaces and `$(touch INJECTED)` is preserved literally; the sentinel file is not created.
- Profile tree hash acceptance and changed-tree rejection both pass. The actual local C v1 profile checkpoint hash matches the pinned value: `a8dfb90341c4a5d8ad72a6fd38f5741489114b24efd4898805d0f63ca0542bf7`.
- `docker buildx build --check` passed with small named-context directory substitutes and reported no warnings.
- `python -m compileall -q scripts src tests` and `git diff --check` passed.
- Full suite: **495 tests, 481 passed, 13 skipped, 1 existing error**: `test_budget_exhaustion_writes_gave_up` cannot read its temporary `relaunch.log`. It reproduced in the pre-change baseline.
- The local environment does not have `torch`, `transformers`, `peft`, `safetensors`, or `accelerate`; no local model-load or inference test was possible.

## Boundary

No official image was built. This A worktree does not contain the large model directories, and the exact official base image was not supplied. The BuildKit check and small-file staging test validate packaging paths and configuration behavior; they do not prove real-weight loading, GPU inference, or offline acceptance. The final official-base image still needs its real model context and platform GPU/offline regression.
