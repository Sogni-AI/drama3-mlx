"""Native NVFP4 experts from plotarmordev/drama-3-NVFP4-Pruned.

The checkpoint stores low-nibble-first E2M1 codes, one E4M3 scale per 16
columns, and one FP32 global scale per expert/projection. Repacking bytes into
uint32 changes no values. As TensorFold's nvfp4_experts.cu does, apply the global
scale to the FP32 dot product. Gate/up round to the input dtype; down stays FP32
until the router weight is applied, then each weighted contribution rounds once.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU


def nvfp4_swiglu(up, gate):
    """TensorFold's Drama epilogue: fp32 SiLU/product, one final bf16 cast."""
    gate32, up32 = gate.astype(mx.float32), up.astype(mx.float32)
    return (gate32 / (1.0 + mx.exp(-gate32)) * up32).astype(gate.dtype)


class NVFP4SwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int,
                 *, output_float32: bool = False):
        super().__init__()
        self.output_float32 = output_float32
        self.weight = mx.zeros((num_experts, output_dims, input_dims // 8), mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, input_dims // 16), mx.uint8)
        self.global_scales = mx.ones((num_experts,), mx.float32)

    def __call__(self, x, indices, sorted_indices=False):
        # MLX's global_scale argument is scalar, but this checkpoint has one per
        # expert. Keeping the dot product in fp32 avoids a bf16 rounding before
        # the expert-specific multiplication (TensorFold applies it before cast).
        out = mx.gather_qmm(
            x.astype(mx.float32), self.weight, self.scales,
            rhs_indices=indices, transpose=True, group_size=16, bits=4,
            mode="nvfp4", sorted_indices=sorted_indices,
        )
        out = out * self.global_scales[indices][..., None, None]
        return out if self.output_float32 else out.astype(x.dtype)


class NVFP4SwitchGLU(SwitchGLU):
    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__(input_dims, hidden_dims, num_experts, activation=nvfp4_swiglu)
        self.gate_proj = NVFP4SwitchLinear(input_dims, hidden_dims, num_experts)
        self.up_proj = NVFP4SwitchLinear(input_dims, hidden_dims, num_experts)
        self.down_proj = NVFP4SwitchLinear(
            hidden_dims, input_dims, num_experts, output_float32=True,
        )
