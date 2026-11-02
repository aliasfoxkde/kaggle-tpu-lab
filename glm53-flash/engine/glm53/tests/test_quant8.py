"""int8 per-channel quantization for the non-expert matrices (`glm53.quant8`).

The interesting properties are: the quantized value must round-trip within the int8 grid, the device
quantizer and the host (NumPy) quantizer must agree exactly, the scale layout must follow the sharding
axis, and the byte accounting must match the arrays actually stored.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from glm53 import quant8

# MIN_SIZE is 1 << 20 elements; a 1024x1024 matrix is exactly at the threshold.
BIG = 1024


def _rand(inp, out, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((inp, out)).astype(np.float32)


# ----------------------------------------------------------------------------- is_q8 / eligible / scale_axis
def test_is_q8_true_only_for_exact_q_s_dicts():
    assert quant8.is_q8({"q": 1, "s": 2})
    assert not quant8.is_q8({"q": 1, "s": 2, "x": 3})
    assert not quant8.is_q8({"q": 1})
    assert not quant8.is_q8({"s": 2})
    assert not quant8.is_q8([1, 2])
    assert not quant8.is_q8(None)


def test_eligible_requires_2d_big_float_array_and_a_partition_spec():
    a = _rand(BIG, BIG)
    assert quant8.eligible("w", a, P("in", "out"))
    # not a PartitionSpec
    assert not quant8.eligible("w", a, ("in", "out"))
    # 1-D
    assert not quant8.eligible("w", a[0], P("in"))
    # too small
    assert not quant8.eligible("w", np.zeros((4, 4), np.float32), P(None, None))
    # skipped keys
    assert not quant8.eligible("router_w", a, P(None, None))
    assert not quant8.eligible("lm_head", a, P(None, None))
    # bad dtype
    assert not quant8.eligible("w", a.astype(np.int8), P(None, None))


def test_eligible_accepts_bf16_and_float32_dtypes():
    a = _rand(BIG, BIG)
    assert quant8.eligible("w", a.astype(jnp.bfloat16), P(None, None))
    assert quant8.eligible("w", jnp.asarray(a), P(None, None))
    assert quant8.eligible("w", a.astype(np.dtype("bfloat16")), P(None, None))
    assert quant8.eligible("w", a.astype(np.dtype("float32")), P(None, None))


def test_scale_axis_follows_the_sharded_dimension():
    assert quant8.scale_axis(P("in", "out")) == 0
    assert quant8.scale_axis(P(None, "out")) == 1
    assert quant8.scale_axis(P(None, None)) == 1
    assert quant8.scale_axis(P()) == 1


# ----------------------------------------------------------------------------- quantize_array
def test_quantize_array_axis0_gives_per_row_scales():
    w = _rand(8, 16, seed=1)
    r = quant8.quantize_array(w, 0)
    assert r["q"].shape == (8, 16)
    assert r["q"].dtype == jnp.int8
    assert r["s"].shape == (8, 1)
    assert r["s"].dtype == jnp.float32
    # the row scale is the row's absmax / 127
    for i in range(8):
        np.testing.assert_allclose(float(r["s"][i, 0]), np.abs(w[i]).max() / 127.0, rtol=1e-6)
    # the extreme value maps to +-127
    assert int(np.abs(np.asarray(r["q"])).max()) == 127


def test_quantize_array_axis1_gives_per_column_scales():
    w = _rand(8, 16, seed=2)
    r = quant8.quantize_array(w, 1)
    assert r["s"].shape == (1, 16)
    for j in range(16):
        np.testing.assert_allclose(float(r["s"][0, j]), np.abs(w[:, j]).max() / 127.0, rtol=1e-6)


def test_quantize_array_roundtrip_within_the_int8_grid():
    w = _rand(32, 32, seed=3)
    r = quant8.quantize_array(w, 0)
    deq = np.asarray(r["q"].astype(jnp.float32) * r["s"])
    err = np.abs(deq - w).max(axis=1)
    # per row: |w| <= amax, so the error is at most amax/254
    for i in range(32):
        assert err[i] <= np.abs(w[i]).max() / 254.0 + 1e-6


def test_quantize_array_clamps_and_survives_an_all_zero_row():
    w = np.zeros((3, 4), np.float32)
    w[1] = 1.0
    r = quant8.quantize_array(w, 0)
    # the zero row gets the 1e-12 floor scale, not a NaN
    assert np.isfinite(np.asarray(r["s"])).all()
    assert float(r["s"][0, 0]) == pytest.approx(1e-12 / 127.0)
    assert np.asarray(r["q"])[0].tolist() == [0, 0, 0, 0]


def test_quantize_array_never_produces_minus_128():
    w = -_rand(16, 16, seed=4)
    r = quant8.quantize_array(w, 0)
    assert int(np.asarray(r["q"]).min()) >= -127


# ----------------------------------------------------------------------------- quantize_array_host
@pytest.mark.parametrize("axis", [0, 1])
def test_host_quantizer_matches_the_device_quantizer(axis):
    w = _rand(64, 48, seed=5)
    dev = quant8.quantize_array(w, axis)
    host = quant8.quantize_array_host(w, axis)
    np.testing.assert_array_equal(np.asarray(dev["q"]), host["q"])
    np.testing.assert_allclose(np.asarray(dev["s"]), host["s"], rtol=1e-6)


def test_host_quantizer_chunking_does_not_change_the_result():
    w = _rand(37, 19, seed=6)
    one = quant8.quantize_array_host(w, 0)
    many = quant8.quantize_array_host(w, 0, rows=5)
    np.testing.assert_array_equal(one["q"], many["q"])
    np.testing.assert_allclose(one["s"], many["s"], rtol=1e-6)


def test_host_quantizer_axis1_needs_two_passes_and_is_column_exact():
    # a column whose extreme value lives in the LAST row: a chunked single-pass implementation
    # would miss it, so the two-pass path must see the whole column.
    w = np.zeros((8, 4), np.float32)
    w[7, 2] = 100.0
    r = quant8.quantize_array_host(w, 1, rows=3)
    assert r["s"].shape == (1, 4)
    np.testing.assert_allclose(r["s"][0, 2], 100.0 / 127.0, rtol=1e-6)
    assert r["q"][7, 2] == 127


def test_host_quantizer_returns_numpy_arrays():
    w = _rand(8, 8, seed=7)
    r = quant8.quantize_array_host(w, 0)
    assert isinstance(r["q"], np.ndarray) and r["q"].dtype == np.int8
    assert isinstance(r["s"], np.ndarray) and r["s"].dtype == np.float32


# ----------------------------------------------------------------------------- quantize_layer
def _layer():
    return {
        "ln1": np.ones(16, np.float32),                       # 1-D: never eligible
        "q_proj": np.full((BIG, BIG), 0.5, np.float32),      # 2-D, big: eligible
        "router_w": np.full((BIG, BIG), 0.5, np.float32),     # skipped key
        "attn": {"o": np.full((BIG, BIG), 0.5, np.float32)},  # nested
    }


def test_quantize_layer_quantizes_only_eligible_leaves_and_returns_matching_specs():
    p = _layer()
    spec = {"ln1": P(None), "q_proj": P("in", "out"), "router_w": P(None), "attn": {"o": P("in", "out")}}
    out, out_s = quant8.quantize_layer(p, spec)
    assert quant8.is_q8(out["q_proj"])
    assert quant8.is_q8(out["attn"]["o"])
    assert not quant8.is_q8(out["router_w"])
    assert not isinstance(out["ln1"], dict)
    # specs mirror the params: a q8 leaf gets a two-entry spec
    assert out_s["q_proj"] == {"q": P("in", "out"), "s": P("in", "out")}
    assert out_s["attn"]["o"] == {"q": P("in", "out"), "s": P("in", "out")}
    # untouched leaves keep their spec
    assert out_s["ln1"] == P(None)
    assert out_s["router_w"] == P(None)


def test_quantize_layer_uses_the_scale_axis_of_the_leaf_spec():
    p = {"w": np.full((BIG, BIG), 0.5, np.float32)}
    out, _ = quant8.quantize_layer(p, {"w": P("in", None)})
    assert out["w"]["s"].shape == (BIG, 1)
    out, _ = quant8.quantize_layer(p, {"w": P(None, "out")})
    assert out["w"]["s"].shape == (1, BIG)


def test_quantize_layer_accepts_a_custom_quantizer():
    calls = []

    def fake(w, axis):
        calls.append((np.asarray(w).shape, axis))
        return {"q": np.zeros(w.shape, np.int8), "s": np.ones((1, 1), np.float32)}

    p = {"w": np.full((BIG, BIG), 0.5, np.float32)}
    out, _ = quant8.quantize_layer(p, {"w": P("in", None)}, fn=fake)
    assert calls == [((BIG, BIG), 0)]
    assert out["w"]["q"].shape == (BIG, BIG)


def test_quantize_layer_leaves_a_non_dict_param_tree_untouched():
    arr = np.full((BIG, BIG), 0.5, np.float32)
    out, spec = quant8.quantize_layer(arr, P("in", "out"))
    assert out is arr
    assert spec == P("in", "out")


# ----------------------------------------------------------------------------- expand_specs
def test_expand_specs_splits_a_q8_leaf_into_q_and_s():
    q8 = {"q": np.zeros((2, 2), np.int8), "s": np.zeros((2, 1), np.float32)}
    out = quant8.expand_specs(q8, P("in", "out"))
    assert out == {"q": P("in", "out"), "s": P("in", "out")}


def test_expand_specs_walks_dicts_and_lists():
    tree = {"a": [{"q": np.zeros((2, 2), np.int8), "s": np.zeros((2, 1), np.float32)}]}
    out = quant8.expand_specs(tree, {"a": [P("in", "out")]})
    assert out["a"][0] == {"q": P("in", "out"), "s": P("in", "out")}


def test_expand_specs_leaves_non_q8_leaves_alone():
    assert quant8.expand_specs(np.zeros(3), P(None)) == P(None)
    assert quant8.expand_specs({"a": np.zeros(3)}, {"a": P(None)}) == {"a": P(None)}


# ----------------------------------------------------------------------------- dequant_tree
def test_dequant_tree_dequantizes_q8_nodes_recursively():
    q = np.array([[127, -127], [0, 1]], np.int8)
    s = np.array([[0.5], [2.0]], np.float32)
    tree = {"w": {"q": q, "s": s}, "list": [{"q": q, "s": s}], "plain": np.zeros(3)}
    out = quant8.dequant_tree(tree, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(out["w"]), q.astype(np.float32) * s)
    np.testing.assert_allclose(np.asarray(out["list"][0]), q.astype(np.float32) * s)
    assert np.asarray(out["plain"]).tolist() == [0.0, 0.0, 0.0]


def test_dequant_tree_default_dtype_is_bfloat16():
    q8 = {"q": np.array([[1]], np.int8), "s": np.array([[2.0]], np.float32)}
    out = quant8.dequant_tree(q8)
    assert out.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(out, np.float32), [[2.0]])


def test_dequant_tree_does_not_mutate_the_input():
    q8: dict[str, np.ndarray] = {"q": np.array([[1]], np.int8), "s": np.array([[2.0]], np.float32)}
    quant8.dequant_tree(q8)
    assert quant8.is_q8(q8)
    assert q8["q"].dtype == np.int8


# ----------------------------------------------------------------------------- count_bytes
def test_count_bytes_counts_q8_as_int8_plus_fp32_scales():
    q = np.zeros((4, 8), np.int8)
    s = np.zeros((4, 1), np.float32)
    assert quant8.count_bytes({"q": q, "s": s}) == 4 * 8 + 4 * 1 * 4


def test_count_bytes_sums_nested_trees():
    q = np.zeros((2, 2), np.int8)
    s = np.zeros((2, 1), np.float32)
    plain = np.zeros((3, 3), np.float32)          # 36 bytes
    total = quant8.count_bytes({"a": {"q": q, "s": s}, "b": [plain, plain]})
    assert total == (2 * 2 + 2 * 1 * 4) + 2 * 36


def test_count_bytes_of_a_scalar_is_zero():
    assert quant8.count_bytes(5) == 0
    assert quant8.count_bytes("x") == 0
