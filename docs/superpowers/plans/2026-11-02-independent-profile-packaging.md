# Independent Profile Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add T6 offline packaging and hash-preflight support for the independent profile checkpoint while preserving the empty-profile default.

**Architecture:** `stage_submission.py` copies the configured profile checkpoint only in independent mode and rewrites its staged path. `cloud_preflight.py` verifies its complete tree hash; the integrated Docker image contains the fixed v1 profile directory and preflights it through the existing independent-profile config.

**Tech Stack:** Python standard library, unittest, BuildKit Dockerfile.

**Spec:** `docs/superpowers/specs/2026-11-02-independent-profile-packaging-design.md` and T6 in `docs/EXECUTION_PLAN_CYCLE2.md`.

## Global Constraints

- Keep `configs/integrated.json` on `profile_mode=empty`.
- Do not add weights to Git; use the recorded C v1 checkpoint tree SHA-256.
- Do not claim an official Docker build unless the confirmed official base image and engine are available.

## Review Focus

- Empty mode must not require or stage a profile checkpoint; independent mode must fail if it is missing.
- A modified file anywhere in the profile checkpoint tree must fail hash preflight.
- Paths containing shell-safe special characters must not be rejected or interpreted by the generated launcher.
- Staged config paths must resolve inside the staged participant directory.
- Docker preflight must actually load the independent-profile config while the runtime default stays empty.

### Task 1: Independent profile staging and offline hash preflight

**Files:**
- Modify: `scripts/stage_submission.py`
- Modify: `scripts/cloud_preflight.py`
- Modify: `Dockerfile.integrated`
- Modify: `configs/integrated_artifacts.json`
- Modify: `docs/CLOUD_QUICKSTART.md`
- Modify: `docs/EXECUTION_PLAN_CYCLE2.md`
- Test: `tests/test_cloud_contract.py`
- Test: `tests/test_cloud_preflight.py`

**Interfaces:**
- Consume the existing `profile_mode`, `profile_model_dir`, `check_packaged_files`, and named `model_assets` context.
- Add a pure `verify_model_hashes(config, manifest)` helper used by the CLI and tests. It raises `ValueError` for a missing profile hash or a tree mismatch in independent mode.

- [x] **Step 1: Write failing tests** for independent staging, blank profile path, shell quoting, profile tree hash acceptance, and changed-tree rejection.
- [x] **Step 2: Run the focused Docker/cloud command** and confirm failures against the old profile staging/hash behavior and current Python-path whitelist.
- [x] **Step 3: Implement the minimum staging, preflight, manifest, and Docker changes**. Keep the default config unchanged.
- [x] **Step 4: Re-run focused tests and verify the actual C v1 checkpoint tree digest** against the pinned value.
- [x] **Step 5: Run the full unittest suite, `python -m compileall -q scripts src tests`, `git diff --check`, and check Docker availability; record the unrelated failure accurately.** `docker buildx build --check` passed; no image was built.
- [x] **Step 6: Commit the T6 implementation and test evidence.**
