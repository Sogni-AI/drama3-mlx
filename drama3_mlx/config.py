"""Drama 3 (fish_omni) checkpoint configuration.

Validation mirrors fish_speech/models/text2semantic/fish_omni/config.py of
fish-speech v3.0.0-beta: every architecture flag this port hard-codes is checked,
so an incompatible checkpoint fails at load time instead of producing noise.
"""

from dataclasses import dataclass, fields
from typing import Any

from mlx_lm.models.qwen3_5 import TextModelArgs

FISH_OMNI_MODEL_TYPE = "fish_omni"
TEXT_MODEL_TYPES = ("fish_omni_text", "qwen3_5_moe_text")
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"
FP8_QUANT_METHOD = "bsgemm-moe"
FP8_BLOCK_SIZE = 128
# (key, default when absent, required value)
FIXED_TEXT_FLAGS = (
    ("rotary_interleaved", False, False),
    ("use_qk_norm", True, True),
    ("attn_output_gate", True, True),
    ("layernorm_zero_centered_gamma", True, True),
    ("shared_expert_gate", True, True),
)
FIXED_AUDIO_DECODER_FLAGS = {
    "decoder_type": "codebook_mtp_shared",
    "rotary_interleaved": True,
    "attention_output_gate": True,
    "layernorm_zero_centered_gamma": True,
    "use_moe": False,
    "mtp_use_swa": False,
    "attention_qkv_bias": False,
    "attention_o_bias": False,
    "dropout": 0.0,
}
# mlx-lm's Qwen3.5 decoder places full attention on every fourth layer, as Drama 3
# does; another layout fails strict weight loading (linear_attn vs self_attn names).
FULL_ATTENTION_INTERVAL = 4


@dataclass(frozen=True, kw_only=True)
class AudioDecoderConfig:
    dim: int
    intermediate_size: int
    n_head: int
    n_local_heads: int
    head_dim: int
    norm_eps: float
    rope_base: float
    rotary_percent: float
    vocab_size: int
    num_codebooks: int
    shared_enrichment_layers: int
    vq_backbone_codebooks: int

    def __post_init__(self) -> None:
        if (
            self.n_head % self.n_local_heads
            or not 0 < self.rotary_dim <= self.head_dim
            or self.rotary_dim % 2
        ):
            raise ValueError("invalid tower attention geometry")
        if self.num_codebooks < 2 or not 0 <= self.vq_backbone_codebooks < self.num_codebooks:
            raise ValueError("invalid codebook counts")
        if self.shared_enrichment_layers < 1:
            raise ValueError("shared_enrichment_layers must be >= 1")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioDecoderConfig":
        for key, value in FIXED_AUDIO_DECODER_FLAGS.items():
            if data.get(key, value) != value:
                raise ValueError(f"unsupported audio_decoder_config.{key}={data[key]!r}")
        if data.get("text_dim", data["dim"]) != data["dim"]:
            raise ValueError("audio_decoder_config.text_dim must equal dim")
        if data.get("n_layer") != data["shared_enrichment_layers"]:
            raise ValueError("n_layer must equal shared_enrichment_layers")
        placement = data.get("vq_feature_placement")
        if placement not in ("backbone_partial", "mtp_shared_pre") or (
            placement == "mtp_shared_pre" and data["vq_backbone_codebooks"] != 0
        ):
            raise ValueError(f"unsupported vq_feature_placement {placement!r}")
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(kw_only=True)
class Drama3Config:
    text: dict[str, Any]
    audio_decoder: AudioDecoderConfig
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    fp8_experts: bool = False

    def __post_init__(self) -> None:
        start, end = self.semantic_start_token_id, self.semantic_end_token_id
        if not 0 <= start <= end < self.text["vocab_size"]:
            raise ValueError("the semantic band must lie inside the text vocabulary")
        if start <= self.eos_token_id <= end:
            raise ValueError("eos_token_id lies inside the semantic band")
        if self.audio_decoder.dim != self.text["hidden_size"]:
            raise ValueError("tower and backbone hidden sizes must match")
        if self.audio_decoder.vocab_size != end - start + 1:
            raise ValueError("tower vocabulary must match the semantic token band")

    @property
    def num_codebooks(self) -> int:
        return self.audio_decoder.num_codebooks

    def text_args(self) -> TextModelArgs:
        return TextModelArgs.from_dict(
            {**self.text, "full_attention_interval": FULL_ATTENTION_INTERVAL}
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Drama3Config":
        if data.get("model_type") != FISH_OMNI_MODEL_TYPE:
            raise ValueError(
                f"model_type={data.get('model_type')!r}; expected {FISH_OMNI_MODEL_TYPE!r}"
            )
        if data.get("audio_encoder_config") is not None:
            raise NotImplementedError("audio_encoder_config is not supported")
        if data.get("speech_template_version") != SPEECH_TEMPLATE_VERSION:
            raise NotImplementedError(
                f"speech_template_version={data.get('speech_template_version')!r}; "
                f"only {SPEECH_TEMPLATE_VERSION} is supported"
            )
        quantization = data.get("quantization_config")
        if quantization is not None and (
            quantization.get("quant_method") != FP8_QUANT_METHOD
            or quantization.get("weight_block_size") != [FP8_BLOCK_SIZE] * 2
        ):
            raise NotImplementedError(
                f"only {FP8_QUANT_METHOD} quantization with weight_block_size "
                f"[{FP8_BLOCK_SIZE}, {FP8_BLOCK_SIZE}] is supported"
            )
        text = validate_text_config(data["text_config"])
        eos = data.get("eos_token_id", data["text_config"].get("eos_token_id"))
        if eos is None:
            raise ValueError("config has no eos_token_id")
        return cls(
            text=text,
            audio_decoder=AudioDecoderConfig.from_dict(data["audio_decoder_config"]),
            semantic_start_token_id=data["semantic_start_token_id"],
            semantic_end_token_id=data["semantic_end_token_id"],
            eos_token_id=eos,
            fp8_experts=quantization is not None,
        )


def validate_text_config(data: dict[str, Any]) -> dict[str, Any]:
    """Check a fish_omni_text / qwen3_5_moe_text config against this port."""
    if data.get("model_type") not in TEXT_MODEL_TYPES:
        raise ValueError(
            f"text_config.model_type={data.get('model_type')!r}; "
            f"expected one of {TEXT_MODEL_TYPES}"
        )
    if any(t != "moe" for t in data.get("mlp_layer_types") or ()):
        raise NotImplementedError("every text_config.mlp_layer_types must be moe")
    for key, default, required in FIXED_TEXT_FLAGS:
        if data.get(key, default) is not required:
            raise NotImplementedError(
                f"text_config.{key} must be {required}; got {data.get(key)!r}"
            )
    if data.get("tie_word_embeddings", False):
        raise NotImplementedError("the backbone keeps a separate lm_head")
    rope = data.get("rope_parameters") or data
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise NotImplementedError("only the default rope_type is supported")
    text = dict(data)
    # Older configs keep the rotary factor at the top level; 0.25 is the HF default.
    text["rope_parameters"] = {
        "type": "default",
        "rope_theta": rope["rope_theta"],
        "partial_rotary_factor": rope.get(
            "partial_rotary_factor", data.get("partial_rotary_factor", 0.25)
        ),
    }
    return text
