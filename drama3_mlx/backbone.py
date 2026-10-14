"""Qwen3.5-MoE text backbone of Drama 3, on mlx-lm's Qwen3.5 decoder.

mlx-lm already implements the hybrid Gated DeltaNet / gated-attention stack. Four
pieces are swapped in to match the fish_omni reference exactly:

* ZeroCenteredRMSNorm keeps the checkpoint's zero-centered gains and applies
  (1 + weight) in fp32, instead of mlx-lm's bf16 weight + 1 shift at load time.
* Drama3MoeBlock routes in fp32 and sums the expert outputs in fp32.
* FP8SwitchGLU runs the release's block-FP8 experts directly, dequantizing only
  the experts a forward selects (no conversion, 41.5 GB resident).
* Drama3GatedDeltaNet l2-normalises q and k as the reference does.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModel
from mlx_lm.models.qwen3_next import Qwen3NextMLP
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

from .config import FP8_BLOCK_SIZE, Drama3Config


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by 1 + weight, computed in fp32."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        gain = 1.0 + self.weight.astype(mx.float32)
        return mx.fast.rms_norm(x.astype(mx.float32), gain, self.eps).astype(x.dtype)


class Drama3GatedDeltaNet(GatedDeltaNet):
    """mlx-lm's Gated DeltaNet with the reference's q/k normalisation, x / sqrt(sum x² + 1e-6).

    mlx-lm writes it as rms_norm(x, eps=1e-6), which puts the epsilon inside the
    mean: 128 times larger on Drama 3's 128-wide heads, and a 1.2e-4 relative
    error at layer 0 against the reference that grows to 4e-2 by layer 39. Only the
    single-sequence path drama3-mlx runs is kept (always a cache, no mask).
    """

    def __call__(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
        batch, length, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        history = self.conv_kernel_size - 1
        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros((batch, history, self.conv_dim), dtype=inputs.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = mx.contiguous(conv_input[:, -history:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(batch, length, heads, dims)
            for t, heads, dims in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        # rms_norm(x, eps / d) * d**-0.5 == x / sqrt(sum x² + eps).
        dims = self.head_k_dim
        q = mx.fast.rms_norm(q, None, 1e-6 / dims) / dims
        k = mx.fast.rms_norm(k, None, 1e-6 / dims) * dims**-0.5
        out, cache[1] = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, cache[1])
        cache.advance(length)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, length, -1))


class FP8SwitchLinear(SwitchLinear):
    """Experts [E, out, in] in block FP8 (e4m3 bytes + fp32 128x128 block scales)."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        nn.Module.__init__(self)
        self.weight = mx.zeros((num_experts, output_dims, input_dims), dtype=mx.uint8)
        self.scale_inv = mx.zeros(
            (
                num_experts,
                output_dims // FP8_BLOCK_SIZE,
                input_dims // FP8_BLOCK_SIZE,
            ),
            dtype=mx.float32,
        )

    def __call__(self, x, indices, sorted_indices=False):
        if not sorted_indices:
            # Decode: dequantize one copy per (token, expert) pair, with no host sync.
            pairs = indices.flatten()
            weight = dequantize_fp8_blocks(self.weight[pairs], self.scale_inv[pairs], x.dtype)
            rhs = mx.arange(pairs.size, dtype=mx.uint32).reshape(indices.shape)
            return mx.gather_mm(x, weight.swapaxes(-1, -2), rhs_indices=rhs)
        # Prefill (SwitchGLU sorts 64+ pairs): dequantize each selected expert once.
        selected = np.unique(np.array(indices))
        weight = dequantize_fp8_blocks(
            self.weight[mx.array(selected)], self.scale_inv[mx.array(selected)], x.dtype
        )
        remap = np.zeros(self.weight.shape[0], dtype=np.uint32)
        remap[selected] = np.arange(len(selected), dtype=np.uint32)
        return mx.gather_mm(
            x,
            weight.swapaxes(-1, -2),
            rhs_indices=mx.array(remap)[indices],
            sorted_indices=True,
        )


class FP8SwitchGLU(SwitchGLU):
    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__(input_dims, hidden_dims, num_experts)
        self.gate_proj = FP8SwitchLinear(input_dims, hidden_dims, num_experts)
        self.up_proj = FP8SwitchLinear(input_dims, hidden_dims, num_experts)
        self.down_proj = FP8SwitchLinear(hidden_dims, input_dims, num_experts)


def dequantize_fp8_blocks(weight: mx.array, scale_inv: mx.array, dtype) -> mx.array:
    """Block FP8 [..., out, in] (uint8) with fp32 scales [..., out/128, in/128] -> dtype."""
    *lead, rows, cols = weight.shape
    row_blocks, col_blocks = scale_inv.shape[-2:]
    blocks = mx.from_fp8(weight, dtype=mx.float32).reshape(
        *lead, row_blocks, FP8_BLOCK_SIZE, col_blocks, FP8_BLOCK_SIZE
    )
    scaled = blocks * scale_inv[..., :, None, :, None]
    return scaled.reshape(*lead, rows, cols).astype(dtype)


# MLX affine quantization group of re-quantized experts.
GROUP_SIZE = 64


class Drama3MoeBlock(nn.Module):
    """Top-k routed experts (fp32 router) plus a sigmoid-gated shared expert."""

    def __init__(self, args, *, fp8: bool):
        super().__init__()
        dim = args.hidden_size
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        switch = FP8SwitchGLU if fp8 else SwitchGLU
        self.switch_mlp = switch(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = Qwen3NextMLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        probabilities = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        k = self.top_k
        indices = mx.argpartition(-probabilities, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(probabilities, indices, axis=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        # The reference hands the experts weights rounded to the compute dtype.
        weights = weights.astype(x.dtype).astype(mx.float32)
        experts = self.switch_mlp(x, indices).astype(mx.float32)
        routed = (experts * weights[..., None]).sum(axis=-2).astype(x.dtype)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return routed + shared


def build_backbone(config: Drama3Config, *, fp8: bool) -> TextModel:
    args = config.text_args()
    model = TextModel(args)
    eps = args.rms_norm_eps
    for layer in model.model.layers:
        layer.input_layernorm = ZeroCenteredRMSNorm(args.hidden_size, eps)
        layer.post_attention_layernorm = ZeroCenteredRMSNorm(args.hidden_size, eps)
        if layer.is_linear:
            layer.linear_attn = Drama3GatedDeltaNet(args)
        else:
            layer.self_attn.q_norm = ZeroCenteredRMSNorm(args.head_dim, eps)
            layer.self_attn.k_norm = ZeroCenteredRMSNorm(args.head_dim, eps)
        layer.mlp = Drama3MoeBlock(args, fp8=fp8)
    model.model.norm = ZeroCenteredRMSNorm(args.hidden_size, eps)
    return model
