"""Load Drama 3 weights: Fish's release (block-FP8 experts) or a converted MLX copy.

The release's experts are kept in FP8 or, by default, re-quantized to 8-bit layer by
layer as they stream in, so memory peaks near the final model size (4-bit loads on a
64 GB Mac). A converted checkpoint's config.json carries {"drama3_mlx": {"version":
1, "experts": {"group_size", "bits", "mode"} or null}}: null keeps bf16 experts.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .backbone import GROUP_SIZE, build_backbone, dequantize_fp8_blocks
from .config import Drama3Config
from .tower import CodebookTower

# Present in the config.json of checkpoints written by drama3_mlx.convert.
MLX_CONFIG_KEY = "drama3_mlx"
MLX_FORMAT_VERSION = 1

EXPERT_KEY = re.compile(
    r"model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)"
)
# How the release's experts are held in memory: block FP8 as published, or bits for
# MLX affine quantization (None = bf16).
EXPERT_FORMATS = {"fp8": "fp8", "bf16": None, "q8": 8, "q4": 4}
RELEASE_PREFIXES = (
    ("model.language_model.", "backbone.model."),
    ("lm_head.", "backbone.lm_head."),
    ("audio_decoder.codebook_embeddings.", "tower.codebook_embeddings."),
    ("audio_decoder_mtp_shared.layers.", "tower.shared_layers."),
    ("audio_decoder_mtp.layers.", "tower.residual_layers."),
    ("audio_decoder_mtp.heads.", "tower.codebook_heads."),
)


class Drama3Model(nn.Module):
    def __init__(self, config: Drama3Config, *, fp8: bool):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config, fp8=fp8)
        self.tower = CodebookTower(config)


def release_name(key: str) -> str:
    for old, new in RELEASE_PREFIXES:
        if key.startswith(old):
            return new + key[len(old) :]
    raise ValueError(f"unexpected checkpoint tensor {key!r}")


def shard_paths(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted({path / name for name in weight_map.values()})
    return [path / "model.safetensors"]


def iter_release_weights(
    path: Path, config: Drama3Config, dtype: mx.Dtype, bits="fp8"
) -> Iterator[tuple[str, mx.array]]:
    """(MLX name, array) for Fish's release, stacking per-expert tensors per layer.

    bits "fp8" keeps the expert bytes and their fp32 block scales as published;
    otherwise each projection's stack is dequantized as soon as both halves are in
    and stored as bf16 (bits None) or MLX affine-quantized (bits 4 / 8). Every other
    tensor is cast to dtype, as the reference does when it loads the backbone.
    """
    num_experts = config.text["num_experts"]
    pending: dict[tuple[str, str, str], dict[int, mx.array]] = defaultdict(dict)
    halves: dict[str, dict[str, mx.array]] = defaultdict(dict)
    for shard in shard_paths(path):
        weights = mx.load(str(shard))
        for key in list(weights):
            value = weights.pop(key)
            match = EXPERT_KEY.fullmatch(key)
            if match is None:
                name = release_name(key)
                if name.endswith("conv1d.weight"):
                    value = value.moveaxis(2, 1)  # torch [C, 1, K] -> MLX [C, K, 1]
                yield name, value.astype(dtype)
                continue
            slot = (match["layer"], match["proj"], match["kind"])
            pending[slot][int(match["expert"])] = value
            if len(pending[slot]) == num_experts:
                experts = pending.pop(slot)
                stacked = mx.stack([experts[e] for e in range(num_experts)])
                mx.eval(stacked)
                stem = f"backbone.model.layers.{match['layer']}.mlp.switch_mlp.{match['proj']}"
                if bits == "fp8":
                    suffix = "weight" if match["kind"] == "weight" else "scale_inv"
                    yield f"{stem}.{suffix}", stacked
                    continue
                halves[stem][match["kind"]] = stacked
                if len(halves[stem]) == 2:
                    pair = halves.pop(stem)
                    yield from requantized(stem, pair["weight"], pair["weight_scale_inv"], bits)
        del weights


def requantized(stem: str, weight: mx.array, scale_inv: mx.array, bits):
    """One expert stack, dequantized from block FP8 to bf16 (bits None) or MLX affine."""
    dense = dequantize_fp8_blocks(weight, scale_inv, mx.bfloat16)
    if bits is None:
        mx.eval(dense)
        yield f"{stem}.weight", dense
        return
    quantized, scales, biases = mx.quantize(dense, group_size=GROUP_SIZE, bits=bits)
    mx.eval(quantized, scales, biases)
    yield f"{stem}.weight", quantized
    yield f"{stem}.scales", scales
    yield f"{stem}.biases", biases


def quantize_experts(model: nn.Module, bits: int) -> None:
    """Swap the model's (bf16) switch_mlp experts for MLX affine-quantized ones."""
    spec = {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}
    nn.quantize(
        model,
        class_predicate=lambda name, module: (
            ".switch_mlp." in name and hasattr(module, "to_quantized") and spec
        ),
    )


def load_model(
    path: str | Path, *, dtype: mx.Dtype = mx.bfloat16, experts: str = "q8"
) -> Drama3Model:
    """experts (fp8, bf16, q8, q4) applies to Fish's release; a converted
    checkpoint keeps the format it was written in."""
    path = Path(path)
    data = json.loads((path / "config.json").read_text())
    config = Drama3Config.from_dict(data)
    meta = data.get(MLX_CONFIG_KEY)
    if meta is None:
        bits = EXPERT_FORMATS[experts]
        model = Drama3Model(config, fp8=bits == "fp8")
        if isinstance(bits, int):
            quantize_experts(model, bits)
        model.load_weights(list(iter_release_weights(path, config, dtype, bits)), strict=True)
        mx.eval(model.parameters())
        return model.eval()
    model = Drama3Model(config, fp8=False)
    if meta["experts"] is not None:
        quantize_experts(model, meta["experts"]["bits"])
    weights = []
    for shard in shard_paths(path):
        weights.extend(mx.load(str(shard)).items())
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model.eval()
