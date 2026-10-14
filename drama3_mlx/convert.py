"""Convert Fish's Drama 3 release into an MLX checkpoint with quantized experts.

The routed experts are 32B of the 35B parameters. They are dequantized from block
FP8 and re-quantized with MLX's affine quantization (or kept in bf16), exactly as
load_model does on the fly; every other tensor stays bf16. A converted copy is for
loading less from disk (25 GB at 4-bit) or for a machine that cannot hold the
release while it re-quantizes. The codec and tokenizer are copied alongside, the
codec as codec.safetensors so that loading never needs the pickle reader.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .codec import read_torch_state_dict
from .backbone import GROUP_SIZE
from .loading import MLX_CONFIG_KEY, MLX_FORMAT_VERSION, load_model

SHARD_BYTES = 5 * 1024**3
COPIED_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "chat_template.jinja", "LICENSE")
# Section IV(a) of the Fish Audio Community License: a Notice file in every copy,
# stating how a Derivative Work changed the Fish Audio Materials.
NOTICE = """These Fish Audio Materials are licensed under the Fish Audio Community License, \
Version 1.0. Copyright \u00a9 39 AI, INC. All Rights Reserved.

Modified by drama3-mlx (not by Fish Audio): converted from fishaudio/drama-3 for
Apple MLX. The routed experts were dequantized from block FP8 and {experts}; all
other weights are unchanged in bf16, and codec.pth was re-saved as
codec.safetensors.

Built with Fish Audio.
"""


def save_shard(dst: Path, number: int, tensors: dict) -> tuple[str, list[str]]:
    name = f"model-{number:05d}.safetensors"
    mx.save_safetensors(str(dst / name), tensors, metadata={"format": "mlx"})
    return name, list(tensors)


def convert(src: Path, dst: Path, *, bits: int | None) -> None:
    """bits None keeps the experts in bf16."""
    data = json.loads((src / "config.json").read_text())
    experts = None if bits is None else {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}
    model = load_model(src, experts="bf16" if bits is None else f"q{bits}")
    dst.mkdir(parents=True, exist_ok=True)

    shard, shard_bytes, shards = {}, 0, []
    for name, value in tree_flatten(model.parameters()):
        shard[name] = value
        shard_bytes += value.nbytes
        if shard_bytes >= SHARD_BYTES:
            shards.append(save_shard(dst, len(shards) + 1, shard))
            shard, shard_bytes = {}, 0
    if shard:
        shards.append(save_shard(dst, len(shards) + 1, shard))

    weight_map = {tensor: name for name, tensors in shards for tensor in tensors}
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1)
    )
    out_config = {k: v for k, v in data.items() if k != "quantization_config"}
    out_config[MLX_CONFIG_KEY] = {"version": MLX_FORMAT_VERSION, "experts": experts}
    (dst / "config.json").write_text(json.dumps(out_config, indent=2))
    mx.save_safetensors(str(dst / "codec.safetensors"), read_torch_state_dict(src / "codec.pth"))
    for name in COPIED_FILES:
        shutil.copy2(src / name, dst / name)
    described = (
        "stored in bf16"
        if experts is None
        else f"re-quantized to {bits}-bit MLX affine quantization (group size {GROUP_SIZE})"
    )
    (dst / "NOTICE").write_text(NOTICE.format(experts=described))

