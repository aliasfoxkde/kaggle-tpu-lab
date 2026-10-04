"""Exact rewrites of slow TPU ops: `model.top_k` (chunked first stage) returns exactly lax.top_k's values and ids at the
engine's shapes (indexer decode / verify, drafter vocab shard), with ties and with the indexer's masked (finfo.min)
scores; `model.gather_scalars` (row gather + lane select) equals the element gather of the latent scales."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from glm53 import model as M  # noqa: E402


@pytest.mark.parametrize("shape,k", [((1, 4, 8192), 128), ((1, 1, 8192), 512), ((1, 3, 19360), 16),
                                     ((2, 5, 1000), 16), ((3, 8193), 64), ((4, 300), 8)])
@pytest.mark.parametrize("mode", ["random", "ties", "masked"])
def test_top_k_equals_lax(shape, k, mode):
    rng = np.random.default_rng(len(shape) * 1000 + k)
    x = rng.standard_normal(shape).astype(np.float32)
    if mode == "ties":
        x = np.round(x, 1)
    if mode == "masked":                                   # most pools invisible: fewer than k real scores
        x[..., rng.random(shape[-1]) < 0.995] = np.finfo(np.float32).min
    a, b = lax.top_k(jnp.asarray(x), k), M.top_k(jnp.asarray(x), k)
    np.testing.assert_array_equal(np.asarray(b[0]), np.asarray(a[0]))
    np.testing.assert_array_equal(np.asarray(b[1]), np.asarray(a[1]))


@pytest.mark.parametrize("B,S,Tq,W", [(1, 32768, 4, 2080), (2, 1024, 1, 96), (1, 1000, 3, 40)])
def test_gather_scalars(B, S, Tq, W):
    rng = np.random.default_rng(S + W)
    a = rng.standard_normal((B, S)).astype(np.float32)
    idx = rng.integers(0, S, (B, Tq, W)).astype(np.int32)
    ref = np.take_along_axis(a[:, None], idx, axis=2)
    np.testing.assert_array_equal(np.asarray(M.gather_scalars(jnp.asarray(a), jnp.asarray(idx))), ref)
