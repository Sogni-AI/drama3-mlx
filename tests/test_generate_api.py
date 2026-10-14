"""Drama3.generate behaviour that needs no reference: accept / attempts and previous_takes."""

import numpy as np
import pytest

from drama3_mlx import Drama3


@pytest.fixture(scope="module")
def tts(tiny_release_tokens):
    return Drama3.load(tiny_release_tokens, experts="fp8")


def test_rejected_chunks_are_sampled_again(tts):
    seen = []

    def accept(segment):
        seen.append(segment.codes.copy())
        return len(seen) == 3

    (segment,) = tts.generate("Hello there.", accept=accept, attempts=5, max_new_tokens=12, seed=0)
    assert len(seen) == 3
    np.testing.assert_array_equal(segment.codes, seen[-1])
    assert not np.array_equal(seen[0], seen[1])


def test_the_last_take_is_kept_when_none_is_accepted(tts):
    seen = []
    (segment,) = tts.generate(
        "Hello there.", accept=lambda s: seen.append(s) and False, attempts=2, max_new_tokens=12, seed=0
    )
    assert len(seen) == 2 and segment is seen[-1]


def test_previous_takes_join_the_prompt(tts):
    first = next(tts.generate("It is due on Wednesday.", max_new_tokens=10, seed=0))
    edit = next(tts.generate(
        "It is due on Thursday.", previous_takes=[(first.text, first.codes)], max_new_tokens=10, seed=0
    ))
    plain = next(tts.generate("It is due on Thursday.", max_new_tokens=10, seed=0))
    assert edit.prompt_rows > plain.prompt_rows + first.codes.shape[1]
