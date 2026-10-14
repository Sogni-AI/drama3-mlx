"""Sampler semantics of fish_omni/tower.py sample_top_k_top_p."""

import mlx.core as mx
import numpy as np

from drama3_mlx.sampling import sample_top_k_top_p

# Probabilities 0.5, 0.3, 0.15, 0.05.
LOGITS = mx.log(mx.array([[0.15, 0.5, 0.05, 0.3]]))


def support(top_k: int, top_p: float, draws: int = 3000) -> set[int]:
    mx.random.seed(0)
    rows = mx.broadcast_to(LOGITS, (draws, 4))
    return set(np.array(sample_top_k_top_p(rows, 1.0, top_k, top_p)).tolist())


def test_top_k_one_is_greedy():
    assert support(top_k=1, top_p=1.0) == {1}


def test_token_crossing_top_p_is_dropped():
    # Cumulative 0.5, 0.8, 0.95, 1.0: with top_p 0.85 the third token crosses it.
    assert support(top_k=30, top_p=0.85) == {1, 3}


def test_highest_ranked_token_is_always_kept():
    assert support(top_k=30, top_p=0.1) == {1}


def test_top_k_limits_ranks():
    assert support(top_k=3, top_p=1.0) == {0, 1, 3}
