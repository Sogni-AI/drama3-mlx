"""Published NVFP4 byte layout, projection arithmetic, and compact expert IDs.

The independent NumPy oracle follows TensorFold b1f4b9b4's
tools/zig/drama_nvfp4_gptq_test.py and nvfp4_experts.cu: low nibble first,
E2M1 times E4M3, FP32 dot product, then the projection's global scale.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.qwen3_5 import TextModelArgs

from drama3_mlx.backbone import Drama3MoeBlock
from drama3_mlx.convert import convert
from drama3_mlx.loading import iter_nvfp4_weights, nvfp4_expert_map
from drama3_mlx.nvfp4 import NVFP4SwitchLinear, nvfp4_swiglu


def decode(codes, scales):
    nibbles = np.stack([codes & 15, codes >> 4], axis=-1).reshape(*codes.shape[:-1], -1)
    table = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    fp4 = table[nibbles & 7] * np.where(nibbles & 8, -1, 1)
    exponent, mantissa = (scales >> 3) & 15, scales & 7
    fp8 = np.where(exponent == 0, mantissa / 8 * 2.0**-6,
                   (1 + mantissa / 8) * 2.0**(exponent.astype(int) - 7))
    return (fp4 * np.repeat(fp8, 16, axis=-1)).astype(np.float32)


@pytest.fixture(params=[mx.cpu, mx.gpu], ids=["cpu", "gpu"])
def device(request):
    previous = mx.default_device()
    mx.set_default_device(request.param)
    yield request.param
    mx.set_default_device(previous)


@pytest.mark.parametrize("sorted_indices", [False, True])
@pytest.mark.parametrize("output_float32", [False, True], ids=["gate-up", "down"])
def test_projection_matches_independent_decode(device, sorted_indices, output_float32):
    rng = np.random.default_rng(47)
    codes = rng.integers(0, 256, (3, 32, 32), dtype=np.uint8)
    scales = rng.integers(1, 100, (3, 32, 4), dtype=np.uint8)
    globals_ = np.array([.013, .71, 1.9], dtype=np.float32)
    linear = NVFP4SwitchLinear(64, 32, 3, output_float32=output_float32)
    linear.weight = mx.array(codes.copy().view(np.uint32))
    linear.scales = mx.array(scales)
    linear.global_scales = mx.array(globals_)
    ids = np.array([0, 0, 1, 2] if sorted_indices else [2, 0, 1, 0], np.uint32)
    x = mx.array(rng.normal(size=(4, 1, 64)), mx.bfloat16)
    x_np = np.array(x.astype(mx.float32))
    expected = (x_np @ decode(codes, scales)[ids].swapaxes(-1, -2)) * globals_[ids, None, None]
    if not output_float32:
        expected = np.array(mx.array(expected).astype(mx.bfloat16).astype(mx.float32))
    output = linear(x, mx.array(ids), sorted_indices)
    assert output.dtype == (mx.float32 if output_float32 else mx.bfloat16)
    actual = np.array(output.astype(mx.float32))
    np.testing.assert_allclose(actual, expected, rtol=.00001 if output_float32 else .008,
                               atol=.00001 if output_float32 else .002)


def test_swiglu_has_one_final_rounding_like_native_epilogue(device):
    rng = np.random.default_rng(71)
    gate = mx.array(np.r_[1.0390625, rng.uniform(-8, 8, 1000)], mx.bfloat16)
    up = mx.array(np.r_[.1953125, rng.uniform(-8, 8, 1000)], mx.bfloat16)
    gate_np, up_np = np.array(gate.astype(mx.float32)), np.array(up.astype(mx.float32))
    expected = gate_np / (1 + np.exp(-gate_np)) * up_np
    expected = np.array(mx.array(expected).astype(mx.bfloat16).astype(mx.float32))
    actual = np.array(nvfp4_swiglu(up, gate).astype(mx.float32))
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == .150390625  # intermediate bf16 SiLU instead yields .1494140625


def test_loader_preserves_original_ids_and_packed_bytes(tmp_path, device):
    ids = [1, 3]
    index = {"metadata": {"drama_experts": {"0": ids}}, "weight_map": {}}
    tensors = {"model.language_model.layers.0.mlp.gate.weight": mx.arange(64).reshape(4, 16)}
    # Deliberately reverse source tensor order. Packed slots follow the index.
    for expert in reversed(ids):
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            stem = f"model.language_model.layers.0.mlp.experts.{expert}.{proj}."
            tensors[stem + "weight"] = mx.full((16, 8), expert, mx.uint8)
            tensors[stem + "weight_scale"] = mx.full((16, 1), 56, mx.uint8)
            tensors[stem + "weight_scale_2"] = mx.array(expert / 100, mx.float32)
    # A projection's companions may reside in different shards.
    partitions = [dict(list(tensors.items())[::2]), dict(list(tensors.items())[1::2])]
    for i, shard in enumerate(partitions):
        name = f"model-{i}.safetensors"
        mx.save_safetensors(str(tmp_path / name), shard)
        index["weight_map"].update({name_: name for name_ in shard})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert nvfp4_expert_map(tmp_path) == {"0": ids}
    actual = dict(iter_nvfp4_weights(tmp_path, {"0": ids}, mx.bfloat16))
    np.testing.assert_array_equal(
        actual["backbone.model.layers.0.mlp.gate.weight"].astype(mx.float32),
        np.arange(64).reshape(4, 16)[ids],
    )
    for proj in ["up_proj", "gate_proj", "down_proj"]:
        stem = f"backbone.model.layers.0.mlp.switch_mlp.{proj}."
        np.testing.assert_array_equal(actual[stem + "weight"][:, 0, 0], [0x01010101, 0x03030303])
        np.testing.assert_allclose(actual[stem + "global_scales"], [.01, .03])
    with pytest.raises(ValueError, match="load directly"):
        convert(tmp_path, tmp_path / "converted", bits=4)
    assert not (tmp_path / "converted").exists()


def test_compact_router_never_selects_pruned_dominant_expert(device):
    args = TextModelArgs(hidden_size=16, num_experts=2, num_experts_per_tok=1,
                         moe_intermediate_size=16, shared_expert_intermediate_size=16)
    block = Drama3MoeBlock(args, fp8=False, nvfp4=True)
    # Full gate has a dominant removed ID 0; compact slots retain IDs 1 and 3.
    full_gate = mx.zeros((4, 16))
    full_gate[0, 0], full_gate[1, 0], full_gate[3, 0] = 100, 1, 2
    block.gate.weight = full_gate[mx.array([1, 3])]
    seen = []
    block.switch_mlp = lambda x, indices: seen.append(indices) or mx.zeros((*indices.shape, 16))
    block.shared_expert = lambda x: mx.zeros_like(x)
    mx.eval(block(mx.ones((1, 1, 16))))
    assert seen[0].item() == 1  # compact slot 1, original expert 3


def test_nvfp4_routing_rounds_weighted_pairs_then_sum(device):
    args = TextModelArgs(hidden_size=16, num_experts=8, num_experts_per_tok=8,
                         moe_intermediate_size=16, shared_expert_intermediate_size=16)
    block = Drama3MoeBlock(args, fp8=False, nvfp4=True)
    rng = np.random.default_rng(5)
    logits = mx.array(rng.normal(size=(1, 1000, 8)), mx.bfloat16)
    down = rng.normal(size=(1, 1000, 8, 16)).astype(np.float32)
    block.gate = lambda x: logits
    block.switch_mlp = lambda x, indices: mx.array(down)
    block.shared_expert = lambda x: mx.zeros_like(x)
    x = mx.zeros((1, 1000, 16), mx.bfloat16)
    # Isolate the combine oracle from CPU/Metal exp implementation differences.
    p = mx.sort(mx.softmax(logits.astype(mx.float32), axis=-1), axis=-1)[..., ::-1]
    p = np.array(p / p.sum(axis=-1, keepdims=True))
    weighted = mx.array(down * p[..., None]).astype(mx.bfloat16).astype(mx.float32)
    expected = np.array(weighted.sum(axis=-2).astype(mx.bfloat16).astype(mx.float32))
    actual = np.array(block(x).astype(mx.float32))
    np.testing.assert_array_equal(actual, expected)


def test_nvfp4_router_ties_use_lower_original_expert_ids(device):
    args = TextModelArgs(hidden_size=16, num_experts=4, num_experts_per_tok=2,
                         moe_intermediate_size=16, shared_expert_intermediate_size=16)
    block = Drama3MoeBlock(args, fp8=False, nvfp4=True)
    block.gate.weight = mx.zeros_like(block.gate.weight)
    seen = []
    block.switch_mlp = lambda x, indices: seen.append(indices) or mx.zeros((*indices.shape, 16))
    block.shared_expert = lambda x: mx.zeros_like(x)
    mx.eval(block(mx.ones((1, 1, 16))))
    np.testing.assert_array_equal(seen[0], [[[0, 1]]])
