"""codec.pth reading and the codec round trip."""

import mlx.core as mx
import numpy as np
import pytest

from drama3_mlx.codec import FishCodec, read_torch_state_dict

REFERENCE_SUMS = """
import json, sys, torch
state = torch.load(sys.argv[2], map_location="cpu", weights_only=True, mmap=True)
print(json.dumps({k: [list(v.shape), str(v.dtype), v.double().sum().item()] for k, v in state.items()}))
"""


@pytest.fixture(scope="module")
def codec_path(checkpoint):
    return checkpoint / "codec.pth"


def test_reader_matches_torch_load(codec_path, run_reference):
    expected = run_reference(REFERENCE_SUMS, str(codec_path))
    state = read_torch_state_dict(codec_path)
    assert set(state) == set(expected)
    for name, (shape, dtype, total) in expected.items():
        array = np.array(state[name].astype(mx.float32))
        assert list(array.shape) == shape, name
        assert str(state[name].dtype).split(".")[-1] == dtype.split(".")[-1], name
        assert np.isclose(array.astype(np.float64).sum(), total, rtol=1e-9, atol=1e-6), name


def test_round_trip_shapes(codec_path):
    codec = FishCodec(codec_path)
    seconds = 1.5
    audio = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(24000 * seconds)))
    codes = codec.encode(audio.astype(np.float32), 24000)
    assert codes.shape == (10, int(np.ceil(seconds * 44100 / codec.frame_length)))
    assert codes.min() >= 0 and codes.max() < 4096
    decoded = codec.decode(codes)
    assert decoded.dtype == np.float32
    assert len(decoded) == codes.shape[1] * codec.frame_length
