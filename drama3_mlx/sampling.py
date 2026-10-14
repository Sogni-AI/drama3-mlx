"""Sampling of Drama 3 frames, as in fish-speech v3.0.0-beta.

Defaults are those of the official Fish Audio SDK (fish_speech sampling.py); the
repetition-aware resampling (RAS) constants are fixed by the training recipe
(fish_omni/tower.py).
"""

import mlx.core as mx

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 30

RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5


def sample_top_k_top_p(
    logits: mx.array, temperature: float, top_k: int, top_p: float
) -> mx.array:
    """One id per row of logits [N, V].

    As in the reference, the token whose cumulative probability crosses top_p is
    dropped too, except that the highest-ranked token is always kept.
    """
    temperature = max(temperature, MIN_TEMPERATURE)
    scaled = logits.astype(mx.float32) / temperature
    order = mx.argsort(-scaled, axis=-1)
    values = mx.take_along_axis(scaled, order, axis=-1)
    ranks = mx.arange(values.shape[-1])
    cumulative = mx.cumsum(mx.softmax(values, axis=-1), axis=-1)
    dropped = ((cumulative > top_p) | (ranks >= top_k)) & (ranks > 0)
    choice = mx.random.categorical(mx.where(dropped, -mx.inf, values), axis=-1)
    return mx.take_along_axis(order, choice[..., None], axis=-1).squeeze(-1)
