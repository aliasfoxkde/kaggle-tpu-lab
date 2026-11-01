# Session History — kaggle-tpu-lab

Running log of agent work sessions on this repo, newest first. Purpose: any
future session (human or agent) can reconstruct what was done, why, and what
remains, without mining Claude transcripts.

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
