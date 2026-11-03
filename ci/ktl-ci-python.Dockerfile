# ktl-ci-python — offline CI image for kaggle-tpu-lab (GitForge pipeline).
# Tests are CPU JAX + torch; everything is baked so jobs never need egress.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "jax[cpu]==0.12.0.1"

RUN pip install --no-cache-dir \
    "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "transformers==5.22.1" \
    "safetensors==0.8.0" \
    "numpy==2.5.4" \
    "pytest==9.2.0" \
    "pytest-cov==7.1.0" \
    "ruff==0.17.3" \
    "mypy==2.4.1"

# pytest-xdist not used in the end: xdist workers retain memory across their
# assigned files and still hit the cgroup cap (2026-11-03: oom_kill on gw0/gw1
# at the 12GiB runner ceiling, twice). The test job instead runs one pytest
# process per test file — a fresh interpreter per file releases XLA compiled
# executables — driven by a shell loop in .gitforge.yml.
RUN pip install --no-cache-dir "pillow==12.0.0"

# flatbuffers: test_aot guards on importorskip; without it the whole file
# silently collects zero tests (rc=5) and the AOT export path is never tested.
RUN pip install --no-cache-dir "flatbuffers==25.9.23"

# gguf: the GGUF reader and IQ-quant tests import it at module scope, so
# without it pytest collection fails before any test runs.
RUN pip install --no-cache-dir "gguf==0.19.0"
