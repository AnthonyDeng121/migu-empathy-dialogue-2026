# T6 Independent Profile Packaging Design

## Intent

Complete A's T6 packaging support for the independent profile checkpoint so an explicitly selected `profile_mode=independent` configuration can be staged and hash-checked offline. Keep the approved `configs/integrated.json` default on empty profiles.

## Approaches

- Bundle a second Dockerfile and staging path for profile runs. This duplicates launcher and preflight behavior.
- Make staging follow the selected config, while the integrated Docker image carries the fixed v1 profile checkpoint as an additional named-context directory and runs preflight against the independent-profile config. This reuses `profile_mode`, `check_packaged_files`, and the existing model-assets context; **selected**.

## Scope and constraints

- Stage the profile checkpoint only when the selected config says `profile_mode=independent`; rewrite its path to the staged relative path.
- Pin the v1 checkpoint tree to the SHA-256 recorded by C and verify it during `cloud_preflight --verify-hashes`.
- Keep runtime's default config and candidate selection unchanged. Do not commit model weights or claim an official image build; tests use small local substitutes.
- Shell-quote Python executable paths instead of rejecting safe characters such as `@` or spaces.

## Verification

Exercise independent-profile staging with small fake checkpoint files, verify preflight accepts the expected tree hash and rejects a changed file, run cloud contract tests and the full unittest suite, and run Docker build only if an official base image and Docker engine are available.
