"""Grouped expert slots (`ResidentFetch._apply_kernel_grouped`: a verify of T tokens of ONE sequence decodes each
distinct expert once for all its tokens): equal to the per-slot kernels (`_apply_kernel`) in interpret mode, f32, for
routings that share experts between tokens and routings that do not, T = 2..4, two quant-type pairs; `apply` takes the
grouped path for a multi-token step of one sequence and the per-slot path for independent rows."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53.resident import ResidentFetch  # noqa: E402
from glm53.tests.test_resident_cpu import N_DEV, random_tables  # noqa: E402


def routing(rng, T, E, k, shared):
    """[T, k] expert ids, distinct within a token; `shared` of them drawn from a small pool all tokens use."""
    pool = rng.choice(E, max(shared, 1) + 2, replace=False)
    rows = []
    for _ in range(T):
        a = rng.choice(pool, shared, replace=False) if shared else np.zeros(0, int)
        rest = rng.choice(np.setdiff1d(np.arange(E), a), k - shared, replace=False)
        rows.append(rng.permutation(np.concatenate([a, rest])))
    return np.stack(rows).astype(np.int32)


@pytest.mark.parametrize("gu,dn", [("IQ2_S", "IQ3_S"), ("IQ3_S", "IQ4_XS")])
@pytest.mark.parametrize("T,shared", [(2, 3), (4, 2), (4, 0), (3, 4)])
def test_grouped_equals_per_slot(gu, dn, T, shared):
    rng = np.random.default_rng(10 * T + shared)
    E, D, mi, k = 12, 256, 2048, 4
    packed, _ = random_tables(rng, E, D, mi, gu, dn)
    p = {key: {pk: jnp.asarray(a[0:1]) for pk, a in packed[key].items()} for key in packed}   # chip 0's planes
    fetch = ResidentFetch({0: {"gate_q": gu, "up_q": gu, "down_q": dn}}, D, mi // N_DEV, out_dtype=jnp.float32,
                          interpret=True)
    fetch.group_min_seq = 2                                                     # (off by default: slower on the v6e)
    idx = jnp.asarray(routing(rng, T, E, k, shared))
    x = jnp.asarray(rng.standard_normal((T, D)).astype(np.float32))
    w = jnp.asarray(rng.random((T, k)).astype(np.float32))
    a = np.asarray(fetch._apply_kernel(x, idx, w, 0, p["gate_q"], p["up_q"], p["down_q"], 10.0))
    b = np.asarray(fetch._apply_kernel_grouped(x, idx, w, 0, p["gate_q"], p["up_q"], p["down_q"], 10.0))
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5 * np.abs(a).max())
    via = np.asarray(fetch.apply(p, x, idx, w, 0, 10.0, seq_tokens=T))          # dispatch: the grouped path
    np.testing.assert_array_equal(via, b)
    rows = np.asarray(fetch.apply(p, x, idx, w, 0, 10.0, seq_tokens=1))         # independent rows: per slot
    np.testing.assert_array_equal(rows, a)
