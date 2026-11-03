# Session History — kaggle-tpu-lab

Running log of agent work sessions on this repo, newest first. Purpose: any
future session (human or agent) can reconstruct what was done, why, and what
remains, without mining Claude transcripts.

## 2026-11-02 → 03 — CI test-job OOM root cause and per-file fix (session a004d908, continued)

Root cause of every `test` job failure (exit 137) found by faithful local
reproduction under the runner's own sandbox cap (12GiB, from
`GITFORGE_SANDBOX_MEMORY_MB=12288` in the runner unit override — NOT the
4GiB code default):

- **Single pytest process OOM-kills at ~45% through the suite** (cgroup
  `oom_kill 1`, SIGKILL). XLA/JAX retains compiled executables for the
  life of the process; resident RSS grows monotonically (1.7G → 12G).
  Reproduced twice at 12G (2026-11-03 run b: killed at 44% in
  test_pallas_mhc).
- **pytest-xdist also OOM-kills** (workers retain memory across their
  file assignments; two runs, `oom_kill` 1-2 each, worker "node down").
- **Fix: one pytest process per test file** — zero OOM kills over the
  whole suite (2h10m), 91% coverage via `--cov-append`.
- Two silent coverage holes surfaced once tests could actually run:
  - `test_vision.py::test_preprocess_layout` needed pillow (now in image);
  - `test_aot.py` collected ZERO tests (rc=5) without flatbuffers — the
    AOT export path was never exercised. Both added to the image (v3).
- `Image.BICUBIC` → `Image.Resampling.BICUBIC` (removed in Pillow 10).
- GitForge ops findings (recorded for the GitForge repo, not this one):
  runner redeployments from other sessions' work killed in-flight CI jobs
  with no retry (12:07/14:54/15:42 on Nov 2); the standalone
  `gitforge@scheduler.service` crash-looped on a missing release binary
  until masked (CI's in-process assigner is the real scheduler — matches
  `gitforge-release-verify`'s `scheduler-owner=ci` contract).
- Push-trigger pipeline YAML gotcha: `FAILED: ` (colon-space) inside a
  plain scalar `run:` breaks the GitForge parser at trigger time with
  "mapping values are not allowed" — the run then never even starts.
  Folded scalar + `FAILED_TEST` fixes it; runs now start on eef0d89
  (run 4e2fc9c3, 2026-11-03T12:14Z).

Image: `ktl-ci-python:3` (adds pillow 12.0.0, flatbuffers 25.9.23;
pytest-xdist intentionally absent). Pipeline: `.gitforge.yml` test step
runs the per-file loop; timeout 8h retained.

## 2026-11-01 — mypy baseline push + full task-list restoration (session 637a1e85)

Context: GitForge run 475e2f07 — lint green, **types red (~154 mypy errors
across 26 files: 55 attr-defined in test stub classes, 20 in tools/, 33 in
engine/production)**, test job cancelled. Prior sessions 2a546700 and
34d99e57 both died on context compaction; their task lists were mined from
the JSONL transcripts and reconciled against repo/git/Kaggle/GitForge state.

Done:
- Recovered the full 22-task list from 4 sessions' transcripts; statuses
  reconciled against evidence (PRs merged, runs in the GitForge store,
  Kaggle kernel status). Chat-only promises flagged.
- mypy fixes in flight (three parallel lanes: tests/, tools/, engine/).
- Local baseline established: lint green; test suite has at least one real
  failure independent of mypy — `test_basestore.py::test_engine_rebuilt_from_store`
  hits transformers 5.22.1 `validate_architecture` ("Expecting NoPE for the
  DSA attention layers, got qk_rope_head_dim=16") from `tiny_config()` in
  test_tiny_vs_hf.py — the fixture needs `qk_rope_head_dim=0` (or an
  upstream-compatible head-dim shape) under the pinned transformers.

Known open items (with evidence):
- GitForge CI green (types job = the mypy work above; test job not yet run
  to completion in CI).
- Full local pytest suite: run started, results pending (prior attempts
  timed out at 120 s — suite needs ~10+ min; first test takes ~48 s).
- Kaggle kernel `qwen38-pr21-uvfix` (upstream PR #21 TPU validation): still
  QUEUED as of 2026-11-01 16:4x UTC.
- No GitHub release/tag exists (only `clean-2026-10-30`); release task open.
- Backlog from 34d99e57, unstarted: upstream re-audit, cold-start
  improvements for existing notebooks, additional notebooks,
  GLM-5.3/Qwen3.8-Flash-Next follow-through (amortyx-stack-extraction.md
  already covers the assessment — verify before re-doing it).

## 2026-11-01 — lint/quality/hygiene sweep (session 2a546700, resumed)

Context: the prior session (34d99e57, 2026-10-31 → 11-01) crashed repeatedly
with "automatic compaction failed" and left ~60 modified + 8 untracked files.
This session restored its task list by mining the JSONL transcript, verified
no orphaned background processes (all 10 subagents finished, one live session),
then resumed task #1.

Done:
- Fixed real bugs found by lint: F821 undefined `a` in bench_endpoint.py
  (NameError on the --image path), B023 loop capture in serve_glm53.py's
  cloudflared reader thread, dead stores `n_in`/`THINK`.
- Whole-repo `ruff check .` green (was 65). pyproject.toml per-file-ignores
  each carry a rationale comment.
- mypy: replaced blanket `strict = true` (2946 errors, mostly JAX-stub noise)
  with relaxed-but-checked baseline + three disabled JAX-fundamental codes;
  documented why per-module strict overrides don't work here (no __init__.py
  in tools/).
- Repo hygiene: untracked `__pycache__` dirs and `.scratch` removed (all
  git-ignored, nothing tracked was deleted); .gitignore already covered them.
- .gitforge.yml pipeline refreshed: whole-repo ruff, mypy on the project
  config, engine tests via pyproject testpaths.
- Aegis re-scan (2922 findings): every CRITICAL/HIGH inspected — all false
  positives (torch `.eval()`, `secrets.token_hex` key generation, README
  placeholders, harness `exec` by design). Dispositions tabled in
  docs/SECURITY-scan.md.
- 4 new test files from the prior session's coverage subagent
  (test_checkpoint_units/hf_convert/quant8/scheduler_units, 1063 lines)
  validated by the local CPU harness venv (/var/tmp/ktl-venv).

Environment notes: system python3 has no jax — tests must run under
/var/tmp/ktl-venv/bin/python (jax 0.12.0.1 CPU; no pytest-xdist in that venv).

## 2026-10-31 → 11-01 — prior session (34d99e57, crashed)

Accomplished before the context death (work is in the uncommitted tree):
- Initial pyproject.toml (ruff + mypy + pytest config) and requirements-ci.txt
- RUF100 autofix stripped needed `# noqa: E402` comments; fixed by adding
  E402 per-file-ignores and rewriting import blocks (partially done at crash).
- Ran the "Raise engine test coverage" and "Lint engine and kernel modules"
  subagents (both finished).
- Wrote docs/SECURITY-scan.md first pass, .gitforge.yml first pass.
- Upstream PR #21 TPU validation kernel `qwen38-pr21-uvfix` left QUEUED on
  Kaggle (capacity-blocked).

## Earlier (see git log f827904..)

Fork integrations audited (docs/UPSTREAM_AUDIT.md), GPU batch lane + Modal
appliance landed, SSRF fix for remote image fetch (#13), qwen38 uv cache pin
(#15).
