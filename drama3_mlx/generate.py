"""Frame generation, as generate_drama3 in fish_omni/generate.py (fish-speech v3.0.0-beta).

Each step draws the main token from the backbone's lm_head (eos or a semantic
token), completes the frame's other codes with the codebook tower, and feeds the
frame back into the backbone.
"""

import numpy as np
import mlx.core as mx

from .loading import Drama3Model
from .prompt import InvalidInputError
from .sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    RAS_WINDOW_FRAMES,
)

PREFILL_CHUNK_SIZE = 128


def run_backbone(model: Drama3Model, frames: mx.array, caches) -> mx.array:
    """Final-norm hidden states [L, H] of frames [L, 1 + N], continuing the caches."""
    backbone, tower = model.backbone, model.tower
    embeddings = tower.embed_backbone(frames, backbone.model.embed_tokens(frames[:, 0]))
    return backbone.model(None, cache=caches, input_embeddings=embeddings[None])[0]


def generate_frames(
    model: Drama3Model,
    prompt: np.ndarray,
    *,
    max_new_tokens: int,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
) -> tuple[np.ndarray, str]:
    """Sample frames after the prompt rows: ([frames, 1 + N] without eos, finish reason).

    The tower cache of prompt row p is filled with frame p + 1 as its label; the
    last prompt row has none yet, so the first decode step writes that slot.
    """
    # The sampler always keeps the top-ranked token, so top_k < 1 would silently
    # turn into greedy decoding.
    if top_k < 1:
        raise InvalidInputError(f"top_k must be at least 1; got {top_k}")
    config, tower = model.config, model.tower
    prompt_rows = len(prompt)
    if max_new_tokens < 1 or prompt_rows + max_new_tokens > config.text["max_position_embeddings"]:
        raise ValueError(f"max_new_tokens={max_new_tokens} does not fit after {prompt_rows} rows")

    caches = model.backbone.make_cache()
    tower_caches = tower.make_caches()
    rows = mx.array(prompt.astype(np.int32))
    for start in range(0, prompt_rows, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prompt_rows)
        hidden = run_backbone(model, rows[start:end], caches)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.forced(
                hidden[: paired_end - start],
                rows[start:paired_end],
                rows[start + 1 : paired_end + 1],
                tower_caches,
            )
        mx.eval(hidden, [c.state for c in caches], [c.state for c in tower_caches])

    hidden = hidden[-1:]
    frame = rows[-1:]
    history = rows[-RAS_WINDOW_FRAMES:, 0]
    produced, finish_reason = [], "length"
    for step in range(max_new_tokens):
        logits = model.backbone.lm_head(hidden)
        frame = tower.decode_step(
            logits,
            hidden,
            frame,
            history,
            tower_caches,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        history = mx.concatenate([history, frame[:, 0]])[-RAS_WINDOW_FRAMES:]
        # Queue the backbone on this frame before waiting on its main token; after
        # an eos the extra step is simply discarded.
        if step + 1 < max_new_tokens:
            hidden = run_backbone(model, frame, caches)
            mx.async_eval(hidden, history)
        main, finite = mx.stack(
            [frame[0, 0], mx.isfinite(logits).all().astype(frame.dtype)]
        ).tolist()
        if not finite:
            raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
        if main == config.eos_token_id:
            finish_reason = "eos"
            break
        produced.append(frame)
    if not produced:
        return np.zeros((0, 1 + config.num_codebooks), dtype=np.int64), finish_reason
    return np.array(mx.concatenate(produced)).astype(np.int64), finish_reason
