"""Codebook tower (codebook_mtp_shared audio decoder) of Drama 3.

Port of fish_speech/models/text2semantic/fish_omni/tower.py (fish-speech
v3.0.0-beta). Per frame the backbone samples the main token (code 0); four shared
enrichment layers then mix the backbone state with code 0, and nine residual
layers each take code i and predict code i + 1. Every tower layer attends over
time through its own KV cache. Rows are [main token id, code 0, ..., code N-1].
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .backbone import ZeroCenteredRMSNorm
from .config import AudioDecoderConfig, Drama3Config
from .sampling import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW_FRAMES
from .sampling import sample_top_k_top_p


class TowerAttention(nn.Module):
    """Gated attention over time; wqkv rows are [Q | Gate | K | V]."""

    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.config = config
        q_width = config.n_head * config.head_dim
        kv_width = config.n_local_heads * config.head_dim
        self.wqkv = nn.Linear(config.dim, 2 * q_width + 2 * kv_width, bias=False)
        self.wo = nn.Linear(q_width, config.dim, bias=False)
        # Always on, whatever audio_decoder_config.attention_qk_norm says (the
        # exporter never writes that field; see the reference TowerAttention).
        self.q_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)
        self.rope = nn.RoPE(config.rotary_dim, traditional=True, base=config.rope_base)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        """x [L, dim] continuing the cached sequence."""
        cfg = self.config
        length = x.shape[0]
        q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
        q, gate, k, v = mx.split(
            self.wqkv(x), [q_width, 2 * q_width, 2 * q_width + kv_width], axis=-1
        )
        q = self.q_norm(q.reshape(1, length, cfg.n_head, cfg.head_dim))
        k = self.k_norm(k.reshape(1, length, cfg.n_local_heads, cfg.head_dim))
        v = v.reshape(1, length, cfg.n_local_heads, cfg.head_dim)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=cfg.head_dim**-0.5,
            mask="causal" if length > 1 else None,
        )
        out = out.transpose(0, 2, 1, 3).reshape(length, q_width)
        return self.wo(out * mx.sigmoid(gate))


class TowerMLP(nn.Module):
    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.attention = TowerAttention(config)
        self.attention_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.feed_forward = TowerMLP(config)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        x = x + self.attention(self.attention_norm(x), cache)
        return x + self.feed_forward(self.ffn_norm(x))


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state, then one pre-norm block."""

    def __init__(self, config: AudioDecoderConfig, *, shared: bool):
        super().__init__()
        self.shared = shared
        embedding = nn.Embedding(config.vocab_size, config.dim)
        # The checkpoint names the embedding differently in the two layer kinds.
        if shared:
            self.semantic_embedding = embedding
        else:
            self.codebook_embedding = embedding
        self.enorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.hnorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerTransformerLayer(config)
        self.final_layernorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.vocab_size = config.vocab_size

    def __call__(
        self,
        hidden: mx.array,
        codes: mx.array,
        semantic: mx.array,
        cache: KVCache,
        shared_input: mx.array | None = None,
    ) -> mx.array:
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        e = embedding(mx.clip(codes, 0, self.vocab_size - 1))
        e = mx.where(semantic[:, None], e, mx.zeros_like(e))
        if shared_input is not None:
            e = e + shared_input
        mixed = self.eh_proj(mx.concatenate([self.enorm(e), self.hnorm(hidden)], axis=-1))
        return self.final_layernorm(self.transformer_layer(mixed, cache))


class CodebookTower(nn.Module):
    def __init__(self, config: Drama3Config):
        super().__init__()
        cfg = config.audio_decoder
        self.cfg = cfg
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        self.codebook_embeddings = nn.Embedding(cfg.vocab_size * cfg.num_codebooks, cfg.dim)
        self.shared_layers = [
            TowerLayer(cfg, shared=True) for _ in range(cfg.shared_enrichment_layers)
        ]
        self.residual_layers = [
            TowerLayer(cfg, shared=False) for _ in range(cfg.num_codebooks - 1)
        ]
        self.codebook_heads = [
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
            for _ in range(cfg.num_codebooks - 1)
        ]
        # The main token is drawn over eos + the semantic band, in id order.
        self._main_candidates = mx.array(
            sorted([self.eos_id, *range(self.semantic_begin, self.semantic_end + 1)]),
            dtype=mx.int32,
        )

    def make_caches(self) -> list[KVCache]:
        return [KVCache() for _ in (*self.shared_layers, *self.residual_layers)]

    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Summed embeddings of codes first..end-1 per frame, accumulated in fp32."""
        weight = self.codebook_embeddings.weight
        if first == end:
            return mx.zeros((frames.shape[0], self.cfg.dim), dtype=weight.dtype)
        offsets = mx.arange(first, end, dtype=frames.dtype) * self.cfg.vocab_size
        rows = frames[:, 1 + first : 1 + end] + offsets
        total = self.codebook_embeddings(rows).astype(mx.float32).sum(axis=1)
        return total.astype(weight.dtype)

    def embed_backbone(self, frames: mx.array, text_embeddings: mx.array) -> mx.array:
        """Backbone input: semantic frames add codes 1..K to the token embedding."""
        count = self.cfg.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0])
        return mx.where(semantic[:, None], mixed, text_embeddings)

    def run_shared(self, hidden, c0, semantic, frames, caches) -> mx.array:
        # The first shared layer also sees the codes the backbone did not.
        unseen = self.embedding_sum(
            frames, 1 + self.cfg.vq_backbone_codebooks, self.cfg.num_codebooks
        )
        unseen = mx.where(
            self.is_semantic(frames[:, 0])[:, None], unseen, mx.zeros_like(unseen)
        )
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(hidden, c0, semantic, caches[index], unseen if index == 0 else None)
        return hidden

    def forced(
        self, hidden: mx.array, frames: mx.array, next_frames: mx.array, caches
    ) -> mx.array:
        """Teacher-force the caches, row p labelled with frame p + 1.

        Returns the head logits [L, N-1, V]; MLX computes them only if evaluated,
        so prefill can drop them for free.
        """
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, frames, caches)
        shared = len(self.shared_layers)
        logits = []
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            hidden = layer(hidden, next_frames[:, index + 1], semantic, caches[shared + index])
            logits.append(head(hidden))
        return mx.stack(logits, axis=1)

    def decode_step(
        self,
        logits: mx.array,
        hidden: mx.array,
        frame: mx.array,
        history: mx.array,
        caches,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> mx.array:
        """Sample the frame [1, 1 + N] after `frame`; history holds recent main tokens."""
        candidates = self._main_candidates
        candidate_logits = logits[:, candidates]
        first = candidates[sample_top_k_top_p(candidate_logits, temperature, top_k, top_p)]
        retry = candidates[
            sample_top_k_top_p(candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)
        ]
        # Repetition-aware sampling: a main token repeating one of the last
        # RAS_WINDOW_FRAMES is redrawn; only audio tokens, never a stop.
        window = history[-RAS_WINDOW_FRAMES:]
        repeated = (window[None, :] == first[:, None]).any(axis=-1)
        repeated = repeated & self.is_semantic(first)
        main = mx.where(repeated, retry, first)
        c0 = mx.clip(main - self.semantic_begin, 0, self.cfg.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, frame, caches)
        codes = [c0]
        shared = len(self.shared_layers)
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            hidden = layer(hidden, codes[-1], semantic, caches[shared + index])
            codes.append(sample_top_k_top_p(head(hidden), temperature, top_k, top_p))
        return mx.stack([main, *codes], axis=-1)
