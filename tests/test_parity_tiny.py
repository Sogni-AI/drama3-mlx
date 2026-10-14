"""The MLX port against Fish's PyTorch reference on a tiny random Drama 3 checkpoint.

Both run in fp32 on the CPU (on M5 GPUs MLX's fp32 GEMM is TF32-like), through
chunked prefill and single-row decode steps, with the tower forced along the rows.
"""

import subprocess

import mlx.core as mx
import numpy as np
import pytest

from drama3_mlx.loading import load_model

from .conftest import TESTS
from .mlx_trace import mlx_trace

PROMPT_ROWS = 150
TOLERANCE = 2e-4  # max |difference| relative to max |reference|


def tiny_rows() -> np.ndarray:
    """Text rows and semantic frames, with prefill spanning two 128-row chunks."""
    rng = np.random.default_rng(1)
    rows = np.zeros((180, 5), dtype=np.int64)
    for position in range(len(rows)):
        if position < 40 or 90 <= position < 100:
            rows[position, 0] = rng.integers(0, 200)
        else:
            codes = rng.integers(0, 64, size=4)
            rows[position, 0] = 256 + codes[0]
            rows[position, 1:] = codes
    return rows


@pytest.fixture(scope="module")
def traces(tmp_path_factory, tiny_checkpoint, ref_python, fish_speech):
    work = tmp_path_factory.mktemp("parity")
    rows = tiny_rows()
    np.save(work / "rows.npy", rows)
    subprocess.run(
        [
            str(ref_python),
            "-I",
            str(TESTS / "reference" / "reference_trace.py"),
            "--fish-speech", str(fish_speech),
            "--checkpoint", str(tiny_checkpoint),
            "--rows", str(work / "rows.npy"),
            "--prompt-rows", str(PROMPT_ROWS),
            "--out", str(work / "reference.npz"),
            "--layers",
        ],
        check=True,
        capture_output=True,
    )
    reference = dict(np.load(work / "reference.npz"))
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = load_model(tiny_checkpoint, dtype=mx.float32, experts="fp8")
        ported = mlx_trace(model, rows, PROMPT_ROWS, layers=True)
    finally:
        mx.set_default_device(previous)
    return reference, ported


@pytest.mark.parametrize(
    "name",
    ["layer_0", "layer_1", "layer_2", "layer_3", "hidden", "main_logits", "tower_logits"],
)
def test_matches_reference(traces, name):
    reference, ported = traces
    expected, actual = reference[name], ported[name]
    assert actual.shape == expected.shape
    error = np.abs(actual - expected).max() / np.abs(expected).max()
    assert error < TOLERANCE, f"{name}: relative error {error:.2e}"
