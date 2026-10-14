"""Write a tiny, randomly initialized checkpoint in the exact Drama 3 format.

The checkpoint has the same tensor names, block-FP8 expert layout and config schema
as fishaudio/drama-3, only with small dimensions, so the MLX port and the
PyTorch reference can be compared layer by layer without the 41.5 GB release.

Needs torch and safetensors (not MLX):

    python tests/reference/make_tiny_checkpoint.py OUT_DIR [--seed N]
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

BLOCK = 128
FP8_MAX = 448.0

TEXT = dict(
    model_type="fish_omni_text",
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=4,
    layer_types=[
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    rope_parameters=dict(
        rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25
    ),
    rms_norm_eps=1e-6,
    max_position_embeddings=4096,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    norm_topk_prob=True,
    mlp_layer_types=["moe"] * 4,
    rotary_interleaved=False,
    use_qk_norm=True,
    attn_output_gate=True,
    layernorm_zero_centered_gamma=True,
    shared_expert_gate=True,
    tie_word_embeddings=False,
)
AUDIO = dict(
    decoder_type="codebook_mtp_shared",
    dim=128,
    text_dim=128,
    intermediate_size=256,
    n_head=4,
    n_local_heads=2,
    head_dim=32,
    norm_eps=1e-6,
    rope_base=10000000.0,
    rotary_percent=0.25,
    rotary_interleaved=True,
    attention_output_gate=True,
    attention_qk_norm=False,
    attention_qkv_bias=False,
    attention_o_bias=False,
    layernorm_zero_centered_gamma=True,
    use_moe=False,
    mtp_use_swa=False,
    dropout=0.0,
    vocab_size=64,
    num_codebooks=4,
    n_layer=2,
    shared_enrichment_layers=2,
    vq_backbone_codebooks=1,
    vq_feature_placement="backbone_partial",
)
SEMANTIC_START = 256
EOS = 200


def config() -> dict:
    return {
        "architectures": ["FishOmniModel"],
        "model_type": "fish_omni",
        "speech_template_version": "reference-bracket-v1",
        "audio_encoder_config": None,
        "text_config": TEXT,
        "audio_decoder_config": AUDIO,
        "semantic_start_token_id": SEMANTIC_START,
        "semantic_end_token_id": SEMANTIC_START + AUDIO["vocab_size"] - 1,
        "eos_token_id": EOS,
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "bsgemm-moe",
            "weight_block_size": [BLOCK, BLOCK],
            "activation_scheme": "dynamic",
        },
    }


def quantize_fp8_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [rows, cols] -> (float8_e4m3fn weight, fp32 scale_inv [rows/128, cols/128])."""
    rows, cols = weight.shape
    blocks = weight.view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale_inv = amax / FP8_MAX
    scaled = blocks / scale_inv[:, None, :, None]
    return scaled.view(rows, cols).to(torch.float8_e4m3fn), scale_inv.float()


def tensors(generator: torch.Generator) -> dict[str, torch.Tensor]:
    def normal(*shape, std=0.05):
        return torch.randn(*shape, generator=generator) * std

    t, a = TEXT, AUDIO
    hidden = t["hidden_size"]
    out: dict[str, torch.Tensor] = {}
    prefix = "model.language_model."
    out[prefix + "embed_tokens.weight"] = normal(t["vocab_size"], hidden, std=0.5)
    out[prefix + "norm.weight"] = normal(hidden, std=0.1)
    out["lm_head.weight"] = normal(t["vocab_size"], hidden, std=0.1)

    key_dim = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    value_dim = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    conv_dim = 2 * key_dim + value_dim
    heads, kv_heads, head_dim = (
        t["num_attention_heads"],
        t["num_key_value_heads"],
        t["head_dim"],
    )
    for index, kind in enumerate(t["layer_types"]):
        layer = f"{prefix}layers.{index}."
        out[layer + "input_layernorm.weight"] = normal(hidden, std=0.1)
        out[layer + "post_attention_layernorm.weight"] = normal(hidden, std=0.1)
        if kind == "linear_attention":
            gdn = layer + "linear_attn."
            out[gdn + "in_proj_qkv.weight"] = normal(conv_dim, hidden, std=0.15)
            out[gdn + "in_proj_z.weight"] = normal(value_dim, hidden, std=0.15)
            out[gdn + "in_proj_b.weight"] = normal(
                t["linear_num_value_heads"], hidden, std=0.15
            )
            out[gdn + "in_proj_a.weight"] = normal(
                t["linear_num_value_heads"], hidden, std=0.15
            )
            out[gdn + "conv1d.weight"] = normal(
                conv_dim, 1, t["linear_conv_kernel_dim"], std=0.4
            )
            out[gdn + "dt_bias"] = 1.0 + normal(t["linear_num_value_heads"], std=0.2)
            out[gdn + "A_log"] = torch.log(
                torch.rand(t["linear_num_value_heads"], generator=generator) * 15 + 1
            )
            out[gdn + "norm.weight"] = 1.0 + normal(t["linear_value_head_dim"], std=0.1)
            out[gdn + "out_proj.weight"] = normal(hidden, value_dim, std=0.15)
        else:
            attn = layer + "self_attn."
            out[attn + "q_proj.weight"] = normal(2 * heads * head_dim, hidden, std=0.15)
            out[attn + "k_proj.weight"] = normal(kv_heads * head_dim, hidden, std=0.15)
            out[attn + "v_proj.weight"] = normal(kv_heads * head_dim, hidden, std=0.15)
            out[attn + "o_proj.weight"] = normal(hidden, heads * head_dim, std=0.15)
            out[attn + "q_norm.weight"] = normal(head_dim, std=0.1)
            out[attn + "k_norm.weight"] = normal(head_dim, std=0.1)
        mlp = layer + "mlp."
        out[mlp + "gate.weight"] = normal(t["num_experts"], hidden, std=0.3)
        out[mlp + "shared_expert_gate.weight"] = normal(1, hidden, std=0.1)
        shared = t["shared_expert_intermediate_size"]
        out[mlp + "shared_expert.gate_proj.weight"] = normal(shared, hidden, std=0.1)
        out[mlp + "shared_expert.up_proj.weight"] = normal(shared, hidden, std=0.1)
        out[mlp + "shared_expert.down_proj.weight"] = normal(hidden, shared, std=0.1)
        inner = t["moe_intermediate_size"]
        for expert in range(t["num_experts"]):
            for proj, shape in (
                ("gate_proj", (inner, hidden)),
                ("up_proj", (inner, hidden)),
                ("down_proj", (hidden, inner)),
            ):
                weight, scale_inv = quantize_fp8_blocks(normal(*shape, std=0.1))
                name = f"{mlp}experts.{expert}.{proj}."
                out[name + "weight"] = weight
                out[name + "weight_scale_inv"] = scale_inv

    dim, vocab = a["dim"], a["vocab_size"]
    q_width, kv_width = a["n_head"] * a["head_dim"], a["n_local_heads"] * a["head_dim"]
    out["audio_decoder.codebook_embeddings.weight"] = normal(
        vocab * a["num_codebooks"], dim, std=0.5
    )

    def tower_layer(name: str, embedding: str) -> None:
        out[f"{name}.{embedding}.weight"] = normal(vocab, dim, std=0.5)
        for norm in ("enorm", "hnorm", "final_layernorm"):
            out[f"{name}.{norm}.weight"] = normal(dim, std=0.1)
        out[f"{name}.eh_proj.weight"] = normal(dim, 2 * dim, std=0.1)
        block = f"{name}.transformer_layer."
        out[block + "attention.wqkv.weight"] = normal(
            2 * q_width + 2 * kv_width, dim, std=0.15
        )
        out[block + "attention.wo.weight"] = normal(dim, q_width, std=0.15)
        out[block + "attention.q_norm.weight"] = normal(a["head_dim"], std=0.1)
        out[block + "attention.k_norm.weight"] = normal(a["head_dim"], std=0.1)
        out[block + "attention_norm.weight"] = normal(dim, std=0.1)
        out[block + "ffn_norm.weight"] = normal(dim, std=0.1)
        out[block + "feed_forward.w1.weight"] = normal(
            a["intermediate_size"], dim, std=0.1
        )
        out[block + "feed_forward.w3.weight"] = normal(
            a["intermediate_size"], dim, std=0.1
        )
        out[block + "feed_forward.w2.weight"] = normal(
            dim, a["intermediate_size"], std=0.1
        )

    for index in range(a["shared_enrichment_layers"]):
        tower_layer(f"audio_decoder_mtp_shared.layers.{index}", "semantic_embedding")
    for index in range(a["num_codebooks"] - 1):
        tower_layer(f"audio_decoder_mtp.layers.{index}", "codebook_embedding")
        out[f"audio_decoder_mtp.heads.{index}.weight"] = normal(vocab, dim, std=0.3)

    # Like the release: FP8 expert weights with fp32 block scales, all else bf16.
    for name, value in out.items():
        if value.dtype == torch.float32 and not name.endswith("weight_scale_inv"):
            out[name] = value.to(torch.bfloat16)
    return out


def write_tokenizer(path: Path) -> None:
    """A word-level tokenizer; the tiny parity checks feed rows directly."""
    from tokenizers import Tokenizer, models

    vocab = {f"t{index}": index for index in range(TEXT["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(path))


def use_release_token_layout() -> None:
    """Switch to the release's vocabulary, semantic band and 10-codebook codec layout,
    so the real tokenizer.json and codec.pth work with the tiny weights."""
    global SEMANTIC_START, EOS
    TEXT["vocab_size"] = 252288
    AUDIO.update(vocab_size=4096, num_codebooks=10)
    SEMANTIC_START, EOS = 248099, 248046


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--release-tokens",
        type=Path,
        metavar="DRAMA3_DIR",
        help="use the release's token layout and copy its tokenizer, codec and licence",
    )
    args = parser.parse_args()
    if args.release_tokens:
        use_release_token_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    weights = tensors(generator)

    # Two shards with an index, like the release.
    names = sorted(weights)
    half = math.ceil(len(names) / 2)
    weight_map = {}
    for shard, chunk in enumerate((names[:half], names[half:]), 1):
        filename = f"model-{shard:05d}-of-00002.safetensors"
        save_file({name: weights[name].contiguous() for name in chunk}, str(args.out_dir / filename))
        weight_map.update({name: filename for name in chunk})
    (args.out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1)
    )
    (args.out_dir / "config.json").write_text(json.dumps(config(), indent=2))
    if args.release_tokens:
        for name in ("tokenizer.json", "codec.pth", "LICENSE"):
            (args.out_dir / name).symlink_to((args.release_tokens / name).resolve())
    else:
        write_tokenizer(args.out_dir / "tokenizer.json")
    print(f"wrote {len(weights)} tensors to {args.out_dir}")


if __name__ == "__main__":
    main()
