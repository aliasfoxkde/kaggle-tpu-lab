"""`glm53.checkpoint` unit tests: fp8/int4 dequantization math, the e4m3fn LUT, weight-map/shard
resolution, and the shard readers — all on synthetic data, no real checkpoint needed."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from glm53 import checkpoint as C


# ----------------------------------------------------------------------------- _fp8_to_f32
def test_fp8_lut_matches_torch():
    u8 = np.arange(256, dtype=np.uint8)
    out = C._fp8_to_f32(u8)
    ref = torch.frombuffer(u8.tobytes(), dtype=torch.float8_e4m3fn).float().numpy()
    finite = np.isfinite(ref)
    assert finite.sum() == 254          # e4m3fn: 0x7F and 0xFF are NaN
    np.testing.assert_allclose(out[finite], ref[finite], rtol=1e-6)
    assert np.isnan(out[~finite]).all()


def test_fp8_lut_known_values():
    out = C._fp8_to_f32(np.array([0x00, 0x80, 0x38, 0xB8, 0x7F], np.uint8))
    assert out[0] == 0.0
    assert out[1] == 0.0 and np.signbit(out[1])          # negative zero
    assert out[2] == pytest.approx(1.0)                 # e=7, m=0
    assert out[3] == pytest.approx(-1.0)
    assert np.isnan(out[4])


def test_fp8_lut_subnormal_range():
    # e=0 -> subnormal: value = sign * (m/8) * 2^-6
    out = C._fp8_to_f32(np.array([0x01, 0x07], np.uint8))
    assert out[0] == pytest.approx(2.0 ** -6 / 8.0)
    assert out[1] == pytest.approx(7 / 8 * 2.0 ** -6)


def test_fp8_lut_is_cached_across_calls():
    C._FP8_LUT = None
    a = C._fp8_to_f32(np.zeros(4, np.uint8))
    assert C._FP8_LUT is not None
    b = C._fp8_to_f32(np.ones(4, np.uint8))
    assert a is not b


# ----------------------------------------------------------------------------- dequant_fp8
def test_dequant_fp8_multiplies_by_the_block_scale():
    w = np.arange(4, dtype=np.float32).reshape(2, 2)
    s = np.array([[1.0, 2.0], [4.0, 8.0]], np.float32)
    out = C.dequant_fp8(w, s, block=1)
    np.testing.assert_allclose(out, [[0.0, 2.0], [8.0, 24.0]])


def test_dequant_fp8_crops_a_scale_grid_larger_than_the_weight():
    w = np.ones((2, 2), np.float32)
    s = np.array([[3.0, 9.0], [5.0, 6.0]], np.float32)
    np.testing.assert_allclose(C.dequant_fp8(w, s, block=128), np.full((2, 2), 3.0))


def test_dequant_fp8_zero_scale_gives_zero():
    w = np.ones((2, 2), np.float32)
    np.testing.assert_allclose(C.dequant_fp8(w, np.zeros((1, 1), np.float32)), np.zeros((2, 2)))


def test_dequant_fp8_device_matches_the_numpy_path():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(3)
    w8 = (rng.standard_normal((128, 256)) * 0.1).astype(np.float32)
    raw = torch.from_numpy(w8).to(torch.float8_e4m3fn)
    u8 = raw.view(torch.uint8).numpy()
    s = np.array([[0.5, 2.0]], np.float32)
    np.testing.assert_allclose(
        np.asarray(C.dequant_fp8_device(jnp.asarray(u8), jnp.asarray(s), block=128)),
        C.dequant_fp8(C._fp8_to_f32(u8), s, block=128),
        rtol=1e-6,
    )


# ----------------------------------------------------------------------------- int4 blockwise
def test_quant_int4_blockwise_packing_layout():
    # block=1 -> one scale per element; the low nibble holds the even column
    w = np.array([[1.0, 56.0], [3.0, 0.0]], np.float32)
    p, s = C.quant_int4_blockwise(w, block=1)
    assert p.shape == (2, 1)
    assert s.shape == (2, 2)
    np.testing.assert_allclose(s, [[1 / 7, 56 / 7], [3 / 7, 1.0]])
    # col0: 1/1 -> q=7 ; col1: 56/8 -> q=7 ; row1: 3/(3/7) -> q=7 ; 0 -> q=0
    assert p[0, 0] == 0x07 | (0x07 << 4)
    assert p[1, 0] == 0x07 | (0x00 << 4)


def test_quant_int4_blockwise_roundtrip_is_on_the_int4_grid():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 384)).astype(np.float32)
    p, s = C.quant_int4_blockwise(w)
    d = C.dequant_int4_numpy(p, s)
    sfull = np.repeat(np.repeat(s, 128, 0), 128, 1)[:256, :384]
    q = d / sfull
    np.testing.assert_allclose(q, np.round(q), atol=1e-4)
    assert q.min() >= -8 and q.max() <= 7


def test_quant_int4_zero_block_gets_unit_scale_not_nan():
    w = np.zeros((128, 256), np.float32)
    p, s = C.quant_int4_blockwise(w)
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s, 1.0)
    assert (p == 0).all()


def test_dequant_int4_numpy_sign_extends_nibbles():
    # 0xF in a nibble means -1 in the int4 grid
    p = np.array([[0xFF]], np.uint8)
    s = np.array([[2.0]], np.float32)
    d = C.dequant_int4_numpy(p, s, block=1)
    assert d.shape == (1, 2)
    np.testing.assert_allclose(d, [[-2.0, -2.0]])


def test_dequant_int4_device_matches_the_numpy_path():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(1)
    w = rng.standard_normal((256, 384)).astype(np.float32)
    p, s = C.quant_int4_blockwise(w)
    a = C.dequant_int4_numpy(p, s)
    b = np.asarray(C.dequant_int4_device(jnp.asarray(p), jnp.asarray(s)))
    np.testing.assert_allclose(a, b, rtol=1e-6)


# ----------------------------------------------------------------------------- weight_map / resolve_shard
@pytest.fixture
def tiny_ckpt(tmp_path):
    tensors = {
        "model.language_model.embed_tokens.weight": torch.eye(4, dtype=torch.float32),
        "model.language_model.norm.weight": torch.ones(4),
        "lm_head.weight": torch.eye(4, dtype=torch.float32),
    }
    fn = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / fn))
    json.dump({"weight_map": dict.fromkeys(tensors, fn)}, open(tmp_path / "model.safetensors.index.json", "w"))
    C.weight_map.cache_clear()
    return tmp_path, fn


def test_weight_map_reads_the_index(tiny_ckpt):
    d, _ = tiny_ckpt
    wm = C.weight_map(str(d))
    assert "model.language_model.embed_tokens.weight" in wm
    assert set(wm.values()) == {"model-00001-of-00001.safetensors"}


def test_weight_map_single_file_model_without_an_index(tmp_path):
    tensors = {"a.weight": torch.ones(1)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    C.weight_map.cache_clear()
    assert C.weight_map(str(tmp_path)) == {"a.weight": "model.safetensors"}


def test_weight_map_second_dir_in_a_colon_list(tmp_path):
    d1, d2 = tmp_path / "one", tmp_path / "two"
    d1.mkdir()
    d2.mkdir()
    tensors = {"x.weight": torch.ones(1)}
    save_file(tensors, str(d2 / "shard-a.safetensors"))
    json.dump({"weight_map": {"x.weight": "shard-a.safetensors"}},
              open(d2 / "model.safetensors.index.json", "w"))
    C.weight_map.cache_clear()
    assert C.weight_map(f"{d1}:{d2}") == {"x.weight": "shard-a.safetensors"}


def test_weight_map_missing_everywhere_raises(tmp_path):
    C.weight_map.cache_clear()
    with pytest.raises(FileNotFoundError):
        C.weight_map(str(tmp_path / "nope"))


def test_resolve_shard_finds_the_dir_that_has_it(tiny_ckpt):
    d, fn = tiny_ckpt
    assert C.resolve_shard(str(d), fn).endswith(fn)
    with pytest.raises(FileNotFoundError):
        C.resolve_shard(str(d), "missing.safetensors")


def test_layer_shards_selects_only_the_requested_layer(tiny_ckpt):
    d, _ = tiny_ckpt
    assert C.layer_shards(str(d), 0) == set()


# ----------------------------------------------------------------------------- ShardReader / RawShardReader
def test_shard_reader_get_release_and_has(tiny_ckpt):
    d, fn = tiny_ckpt
    r = C.ShardReader(str(d))
    assert r.has("lm_head.weight") and not r.has("nope")
    a = r.get("lm_head.weight")
    np.testing.assert_allclose(a, np.eye(4))
    r.release(fn)
    assert fn not in r._open
    np.testing.assert_allclose(r.get("lm_head.weight"), a)   # reopened lazily


def test_raw_reader_matches_the_torch_reader_on_fp8(tmp_path):
    torch.manual_seed(1)
    w8 = (torch.randn(128, 256) * 0.1).to(torch.float8_e4m3fn)
    scale = torch.rand(1, 2) * 0.01 + 0.001
    tensors = {"w": w8, "w_scale_inv": scale, "b": torch.randn(3)}
    save_file(tensors, str(tmp_path / "s.safetensors"))
    json.dump({"weight_map": dict.fromkeys(tensors, "s.safetensors")},
              open(tmp_path / "model.safetensors.index.json", "w"))
    C.weight_map.cache_clear()
    ref = C.ShardReader(str(tmp_path))
    raw = C.RawShardReader(str(tmp_path))
    np.testing.assert_allclose(raw.get("w"), ref.get("w"), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(raw.get("b"), ref.get("b"), rtol=1e-6)
    u8, s = raw.get_fp8_raw("w")
    assert u8.dtype == np.uint8 and u8.shape == (128, 256)
    np.testing.assert_allclose(s, scale.numpy(), rtol=1e-6)


def test_raw_reader_get_fp8_raw_asserts_on_non_fp8(tiny_ckpt):
    d, _ = tiny_ckpt
    r = C.RawShardReader(str(d))
    with pytest.raises(AssertionError):
        r.get_fp8_raw("lm_head.weight")


def test_raw_reader_unsupported_dtype_raises(tmp_path):
    tensors = {"i": torch.arange(4, dtype=torch.int32)}
    save_file(tensors, str(tmp_path / "s.safetensors"))
    json.dump({"weight_map": {"i": "s.safetensors"}}, open(tmp_path / "model.safetensors.index.json", "w"))
    C.weight_map.cache_clear()
    r = C.RawShardReader(str(tmp_path))
    with pytest.raises(ValueError, match="i"):
        r.get("i")
