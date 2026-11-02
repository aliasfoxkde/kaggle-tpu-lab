"""HF-checkpoint conversion (`glm53.hf_convert`).

These converters define the layout contract between a torch `Glm5NextTextModel` and the JAX param tree
in `glm53.model`: torch Linear weights are [out, in] and must be transposed to JAX [in, out]; the KDA
conv1d weight [3C, 1, K] must be split into the q/k/v thirds; the MoE expert stacks must be transposed to
[experts, D, ...].  The fp8 dequantizer must apply a per-128x128-block scale.
"""
from typing import Any

import numpy as np
import pytest
import torch

from glm53 import hf_convert


class _Stub:
    """Duck-typed stand-in for a torch submodule: any attribute may be set, and reading an unset
    attribute raises AttributeError so `hasattr`/`getattr` probes behave like the real thing."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # explicit so mypy permits arbitrary attribute writes on the stub
        object.__setattr__(self, name, value)


def _lin(out, inp):
    m = torch.nn.Linear(inp, out, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.arange(out * inp, dtype=torch.float32).reshape(out, inp))
    return m


# ----------------------------------------------------------------------------- _lin / _np
def test_lin_transposes_torch_linear_to_jax_layout():
    t = _lin(3, 5).weight                      # torch [out=3, in=5]
    out = hf_convert._lin(t)
    assert out.shape == (5, 3)                # JAX [in, out]
    np.testing.assert_array_equal(out, t.detach().numpy().T)


def test_np_upcasts_bfloat16_to_float32():
    t = torch.ones(4, dtype=torch.bfloat16)
    out = hf_convert._np(t)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.ones(4, np.float32))


def test_np_keeps_float32():
    t = torch.arange(3, dtype=torch.float32)
    np.testing.assert_array_equal(hf_convert._np(t), np.arange(3, dtype=np.float32))


# ----------------------------------------------------------------------------- convert_hc
def test_convert_hc_returns_fn_base_scale():
    mod = _Stub()
    mod.fn = _lin(4, 6).weight
    mod.base = torch.arange(4, dtype=torch.float32)
    mod.scale = torch.arange(4, dtype=torch.float32) + 1
    p = hf_convert.convert_hc(mod)
    assert set(p) == {"fn", "base", "scale"}
    assert p["fn"].shape == (6, 4)
    np.testing.assert_array_equal(p["base"], np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(p["scale"], np.arange(4, dtype=np.float32) + 1)


# ----------------------------------------------------------------------------- convert_kda
class _Conv1d:
    def __init__(self, w):
        self.weight = w


def _kda(C=4, K=3, D=8):
    m = _Stub()
    m.qkv_dim = C
    # torch conv1d weight: [3C, 1, K]; rows 0..C-1 = q, C..2C-1 = k, 2C.. = v
    w = torch.arange(3 * C * K, dtype=torch.float32).reshape(3 * C, 1, K)
    m.conv1d = _Conv1d(w)
    m.q_proj = _lin(D, C)
    m.k_proj = _lin(D, C)
    m.v_proj = _lin(D, C)
    m.forget_gate = _Stub()
    m.forget_gate.f_a_proj = _lin(D, C)
    m.forget_gate.f_b_proj = _lin(D, C)
    m.forget_gate.dt_bias = torch.arange(C, dtype=torch.float32)
    m.forget_gate.A_log = torch.arange(C, dtype=torch.float32)
    m.b_proj = _lin(D, C)
    m.g_a_proj = _lin(D, C)
    m.g_b_proj = _lin(D, C)
    m.o_norm = _Stub()
    m.o_norm.weight = torch.ones(C, dtype=torch.float32)
    m.o_proj = _lin(D, D)
    return m


def test_convert_kda_splits_the_conv1d_weight_into_q_k_v():
    C, K = 4, 3
    p = hf_convert.convert_kda(_kda(C, K))
    w = np.arange(3 * C * K, dtype=np.float32).reshape(3 * C, 1, K)[:, 0, :]
    np.testing.assert_array_equal(p["conv_q"], w[:C])
    np.testing.assert_array_equal(p["conv_k"], w[C:2 * C])
    np.testing.assert_array_equal(p["conv_v"], w[2 * C:])


def test_convert_kda_shapes_and_gate_vectors():
    p = hf_convert.convert_kda(_kda())
    assert p["q"].shape == (4, 8)
    assert p["o"].shape == (8, 8)
    assert p["conv_q"].shape == (4, 3)
    assert p["dt_bias"].shape == (4,)
    assert p["A_log"].shape == (4,)
    assert p["o_norm"].shape == (4,)
    assert set(p) == {"q", "k", "v", "conv_q", "conv_k", "conv_v", "f_a", "f_b", "dt_bias", "A_log",
                      "b", "g_a", "g_b", "o_norm", "o"}


# ----------------------------------------------------------------------------- convert_mla
def _mla(has_indexer=True):
    m = _Stub()
    m.q_a_proj = _lin(8, 6)
    m.q_a_layernorm = _Stub()
    m.q_a_layernorm.weight = torch.ones(6, dtype=torch.float32)
    m.q_b_proj = _lin(8, 4)
    m.kv_a_proj_with_mqa = _lin(3, 6)
    m.kv_a_layernorm = _Stub()
    m.kv_a_layernorm.weight = torch.ones(3, dtype=torch.float32)
    m.kv_b_proj = _lin(8, 8)
    m.o_proj = _lin(8, 8)
    if has_indexer:
        ix = _Stub()
        ix.wq_b = _lin(8, 6)
        ix.wk = _lin(8, 8)
        ix.k_norm = _Stub()
        ix.k_norm.weight = torch.ones(8, dtype=torch.float32)
        ix.k_norm.bias = torch.zeros(8, dtype=torch.float32)
        ix.weights_proj = _lin(1, 8)
        ix.index_kpool_compress_ape = torch.ones(8, dtype=torch.float32)
        ix.index_kpool_compress_gate = _lin(8, 8).weight
        m.indexer = ix
    else:
        m.indexer = None
    return m


def test_convert_mla_without_indexer_has_no_indexer_key():
    p = hf_convert.convert_mla(_mla(has_indexer=False))
    assert "indexer" not in p
    assert p["q_a"].shape == (6, 8)
    assert p["kv_a"].shape == (6, 3)
    assert p["q_a_norm"].shape == (6,)
    assert p["kv_a_norm"].shape == (3,)


def test_convert_mla_with_indexer_carries_all_indexer_fields():
    p = hf_convert.convert_mla(_mla())
    ix = p["indexer"]
    assert set(ix) == {"wq_b", "wk", "k_norm_w", "k_norm_b", "weights_proj", "ape", "gate"}
    assert ix["wq_b"].shape == (6, 8)
    assert ix["wk"].shape == (8, 8)
    assert ix["weights_proj"].shape == (8, 1)
    assert ix["ape"].shape == (8,)


# ----------------------------------------------------------------------------- convert_mlp
def test_convert_mlp_transposes_gate_up_down():
    m = _Stub()
    m.gate_proj = _lin(4, 6)
    m.up_proj = _lin(4, 6)
    m.down_proj = _lin(6, 4)
    p = hf_convert.convert_mlp(m)
    assert set(p) == {"gate", "up", "down"}
    assert p["gate"].shape == (6, 4)
    assert p["up"].shape == (6, 4)
    assert p["down"].shape == (4, 6)


# ----------------------------------------------------------------------------- convert_moe
def test_convert_moe_transposes_the_expert_stacks():
    E, D, mi = 2, 4, 3
    m = _Stub()
    m.experts = _Stub()
    # torch layout: gate_up_proj [E, 2mi, D], down_proj [E, D, mi]
    m.experts.gate_up_proj = torch.arange(E * 2 * mi * D, dtype=torch.float32).reshape(E, 2 * mi, D)
    m.experts.down_proj = torch.arange(E * D * mi, dtype=torch.float32).reshape(E, D, mi)
    m.gate = _lin(E, D)
    m.gate.e_score_correction_bias = torch.arange(E, dtype=torch.float32)
    m.shared_experts = _Stub()
    m.shared_experts.gate_proj = _lin(4, D)
    m.shared_experts.up_proj = _lin(4, D)
    m.shared_experts.down_proj = _lin(D, 4)
    p = hf_convert.convert_moe(m)
    assert p["gate_up"].shape == (E, D, 2 * mi)
    assert p["down"].shape == (E, mi, D)
    assert p["router_w"].shape == (D, E)
    assert p["router_bias"].shape == (E,)
    assert set(p["shared"]) == {"gate", "up", "down"}
    # the transpose really happened
    np.testing.assert_array_equal(p["gate_up"][0], np.transpose(m.experts.gate_up_proj.numpy()[0], (1, 0)))
    np.testing.assert_array_equal(p["down"][1], m.experts.down_proj.numpy()[1].T)


# ----------------------------------------------------------------------------- dequant_fp8
def test_dequant_fp8_applies_the_block_scale():
    # block=2 so that the 2x4 weight maps exactly onto the 1x2 scale grid
    w = np.arange(8, dtype=np.float32).reshape(2, 4)
    scale = np.array([[0.5, 2.0]], np.float32)          # [ceil(O/b), ceil(I/b)]
    out = hf_convert.dequant_fp8(w, scale, block=2)
    assert out.dtype == np.float32
    expect = w * np.array([[0.5, 0.5, 2.0, 2.0], [0.5, 0.5, 2.0, 2.0]])
    np.testing.assert_allclose(out, expect)


def test_dequant_fp8_crops_a_scale_grid_larger_than_the_weight():
    # the checkpoint's real case: a 128-block scale grid cropped down to a small tensor
    w = np.ones((2, 2), np.float32)
    scale = np.array([[0.5, 2.0], [4.0, 8.0]], np.float32)
    out = hf_convert.dequant_fp8(w, scale, block=128)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.full((2, 2), 0.5))


def test_dequant_fp8_raises_when_the_scale_grid_cannot_cover_the_weight():
    w = np.ones((5, 7), np.float32)
    with pytest.raises(AssertionError):
        hf_convert.dequant_fp8(w, np.ones((1, 1), np.float32), block=4)


def test_dequant_fp8_zero_scale_block_gives_zero():
    w = np.ones((2, 2), np.float32)
    scale = np.zeros((1, 1), np.float32)
    np.testing.assert_allclose(hf_convert.dequant_fp8(w, scale), np.zeros((2, 2)))
