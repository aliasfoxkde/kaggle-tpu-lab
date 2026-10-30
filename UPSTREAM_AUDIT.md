# Upstream issues and pull request audit

Reviewed 2026-10-30 against `ARahim3/kaggle-tpu-lab` `main`. The consolidated branch is
`clean/kaggle-tpu-lab-2026-10`; it contains the upstream fixes listed here where they
apply, plus the separately reviewable changes in upstream PRs #21, #22, and #23.

## Fixes included in the clean branch

| Upstream item | Finding and disposition |
| --- | --- |
| Issue #20 | The supplied log shows the Kaggle weights on `FUSE.GCSFUSE`, automatic prefetch disabled, and the loader still at `0/18` shards after repeated startup attempts. The Qwen command now explicitly uses vLLM's `prefetch` safetensors strategy. Upstream PR [#21](https://github.com/ARahim3/kaggle-tpu-lab/pull/21). |
| Issue #18 | The GLM ready event sent its bearer key through ntfy. The key is now omitted from relay events, local launcher state is mode `0600`, diagnostic GET routes require the key, key checks are constant-time, and the GCP SSH-forward server binds to loopback. Upstream PR [#22](https://github.com/ARahim3/kaggle-tpu-lab/pull/22). |
| Issue #17 | Queue/timeout errors after SSE headers previously wrote a second HTTP response into the stream. OpenAI and Anthropic streams now receive protocol-shaped error events; non-streaming requests retain HTTP 429/503. The CPU harness now exercises queue saturation for both protocols. Upstream PR [#23](https://github.com/ARahim3/kaggle-tpu-lab/pull/23). |
| Existing PR #9 | Its useful fixes were ported and reconciled with current `main`: Internet preflight, removal of broken single-host TPU metadata variables, clearer engine-exit diagnostics, and KYC guidance. The original PR conflicts with current `main` and remains open upstream. |

## Workarounds and remaining validation

| Upstream item | Status |
| --- | --- |
| Issues #10 and #14 | The attached #10 log confirms vLLM 0.28.0 crashes in `update_draft_token_ids_in_output` with `AttributeError: __delitem__` when MTP and async scheduling are combined. The launcher already supports `--no-async-scheduling` while retaining `--mtp 3`; this is documented and the exact command was added to the issue discussion. The requested Hermes throughput and live TPU behavior remain unverified. |
| Issue #20 backend split | The same log shows the old `tpu-inference` package falling back to vLLM's PyTorch model implementation for this architecture. This patch addresses the directly observed GCSFuse load stall. Migrating the full recipe to `vllm-torchtpu` is a separate runtime change and was not claimed or bundled without a Kaggle TPU validation run. |
| Issue #11 | A GCP TPU shell setup already exists under `gcp/`. Terraform provisioning is still a feature request and needs a concrete deployment/cleanup contract before it can be safely automated. |
| Issue #12 | A community Android companion link is a maintainer-curated README decision, so it is not included in this technical fix set. |
| Issue #15 | “More Models” does not specify a model, dataset, or performance target; no reproducible change can be made yet. |
| Issue #19 | The playlist guest-mode report does not map to a playlist component in this repository. |

## Open pull request review

- PR #9 contains relevant fixes for issues #1, #3, #5, #6, #7, and #8. Its branch conflicts with current `main`; the applicable changes are reconciled in this clean branch.
- PR #4 changes the default model to an abliterated variant and adds a P100 recipe. `CONTRIBUTING.md` explicitly excludes default-model changes and other platform ports from this repository's merge scope, so it is not included.
- PRs #21, #22, and #23 are open against upstream `main`. Their focused changes are also integrated in this branch for a usable consolidated snapshot.

## Validation limits

Passed locally: Python syntax checks, shell syntax, both notebook JSON parses, exact notebook/kernel source parity, focused contracts for prefetch, MTP scheduling flags, TPU metadata sanitization, API-key handling, state-file permissions, and provider-shaped SSE overload errors. The isolated CPU JAX/NumPy/Transformers harness completed with `HARNESS OK`, including tool streams, disconnect handling, context reuse, API-key-protected diagnostics, and both OpenAI and Anthropic queue-saturation SSE behavior. Atheon reported no findings in the changed lines.

The repository has no configured GitHub Actions checks. Kaggle authentication now succeeds using the protected `~/.kaggle/access_token`; quota showed 16.66 TPU hours remaining. Kaggle refused a second TPU kernel because the account already had one queued session (`burst-job-46`), which was left untouched. Therefore no issue-20 checkpoint-load run was started. The live GCSFuse prefetch behavior and Hermes MTP throughput still require a Kaggle runtime run after that queued job finishes.
