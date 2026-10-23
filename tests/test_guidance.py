"""Guidance keeps a separate text-only backbone cache and follows sampled frames."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from drama3_mlx.generate import generate_frames
import drama3_mlx.generate as generation


@pytest.mark.parametrize("gamma,expected_caches", [(1.0, 1), (1.5, 2)])
def test_guidance_tracks_frames_without_unconditional_tower(monkeypatch, gamma, expected_caches):
    caches, calls, draws, forced = [], [], [], []

    def make_cache():
        cache = [SimpleNamespace(state=[])]
        caches.append(cache)
        return cache

    def run_backbone(model, rows, cache):
        branch = next(i for i, c in enumerate(caches) if c is cache)
        calls.append((branch, np.array(rows)))
        return mx.tile(mx.array([[2., 6., 1.]]) if branch == 0 else mx.array([[4., 2., 3.]]), (len(rows), 1))

    def draw(logits, hidden, frame, history, cache, **kwargs):
        draws.append((np.array(logits), np.array(hidden)))
        return mx.array([[7 if len(draws) < 3 else 9, 2, 3]])

    model = SimpleNamespace(
        config=SimpleNamespace(text={"max_position_embeddings": 100}, num_codebooks=2, eos_token_id=9),
        backbone=SimpleNamespace(make_cache=make_cache, lm_head=lambda hidden: hidden),
        tower=SimpleNamespace(make_caches=lambda: [SimpleNamespace(state=[])],
                              forced=lambda *args: forced.append(args), decode_step=draw),
    )
    monkeypatch.setattr(generation, "run_backbone", run_backbone)
    conditional = np.array([[1, 0, 0], [6, 2, 3]])
    plain = np.array([[2, 0, 0]])
    frames, finish = generate_frames(model, conditional, max_new_tokens=5,
                                    unconditioned_prompt=plain, cfg_gamma=gamma)
    assert finish == "eos" and len(frames) == 2
    assert len(caches) == expected_caches
    assert len(forced) == 1  # only the conditioned tower prefill
    expected = [[2., 6., 1.]] if gamma == 1 else [[1., 8., 0.]]
    for logits, hidden in draws:
        np.testing.assert_array_equal(logits, expected)
        np.testing.assert_array_equal(hidden, [[2., 6., 1.]])
    if gamma != 1:
        unconditioned = [rows for branch, rows in calls if branch == 1]
        np.testing.assert_array_equal(unconditioned[0], plain)
        # The unconditional backbone follows the exact frames the tower chose.
        conditioned = [rows for branch, rows in calls if branch == 0]
        assert len(unconditioned) == len(conditioned)
        for plain_rows, conditioned_rows in zip(unconditioned[1:], conditioned[1:]):
            np.testing.assert_array_equal(plain_rows, conditioned_rows)


def test_no_reference_prompt_does_not_allocate_guidance_cache(monkeypatch):
    # This rule lives in the high-level API; the author's built-in voice has γ=1
    # even when a server was configured with γ=1.5.
    from drama3_mlx.model import Drama3
    import drama3_mlx.model as model_module
    calls = []
    tts = object.__new__(Drama3)
    tts.model = SimpleNamespace(config=SimpleNamespace(semantic_start_token_id=10, semantic_end_token_id=20))
    tts.tokenizer = None
    tts.sample_rate = 44100
    tts.codec = SimpleNamespace(decode=lambda codes: np.zeros(4))
    monkeypatch.setattr(model_module, "build_prompt", lambda *args: np.array([[1, 0, 0]]))
    monkeypatch.setattr(Drama3, "_chunk_prompt", lambda *args: np.array([[1, 0, 0]]))

    def generate(model, prompt, **kwargs):
        calls.append(kwargs)
        return np.array([[11, 1, 2]]), "eos"

    monkeypatch.setattr(model_module, "generate_frames", generate)
    list(tts.generate("Plain speech.", cfg_gamma=1.5))
    assert calls[0]["unconditioned_prompt"] is None
