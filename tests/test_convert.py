"""Converted checkpoints load and match the release they came from."""

import json
import shutil

import numpy as np
import pytest

from drama3_mlx.convert import COPIED_FILES, convert
from drama3_mlx.loading import load_model

from .mlx_trace import mlx_trace
from .test_parity_tiny import PROMPT_ROWS, tiny_rows


@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory, tiny_checkpoint, checkpoint):
    """The tiny checkpoint with the release's codec and side files."""
    release = tmp_path_factory.mktemp("release") / "drama-3-tiny"
    shutil.copytree(tiny_checkpoint, release)
    (release / "codec.pth").symlink_to(checkpoint / "codec.pth")
    for name in COPIED_FILES:
        if not (release / name).exists():
            shutil.copy2(checkpoint / name, release / name)
    return release


def trace(path, experts="q8"):
    return mlx_trace(load_model(path, experts=experts), tiny_rows(), PROMPT_ROWS)


@pytest.fixture
def out_dir(tmp_path):
    """Removed after the test: every conversion writes a 1.7 GB codec copy."""
    yield tmp_path / "converted"
    shutil.rmtree(tmp_path / "converted", ignore_errors=True)


def test_bf16_experts_equal_fp8_release(tiny_release, out_dir):
    convert(tiny_release, out_dir, bits=None)
    expected, actual = trace(tiny_release, "fp8"), trace(out_dir)
    for name in ("hidden", "main_logits", "tower_logits"):
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_experts_load(tiny_release, out_dir, bits):
    convert(tiny_release, out_dir, bits=bits)
    config = json.loads((out_dir / "config.json").read_text())
    assert config["drama3_mlx"]["experts"]["bits"] == bits
    assert "quantization_config" not in config
    assert "Built with Fish Audio" in (out_dir / "NOTICE").read_text()
    # A converted copy is exactly what loading the release with q4 / q8 gives.
    expected, actual = trace(tiny_release, f"q{bits}"), trace(out_dir)
    for name in ("hidden", "main_logits", "tower_logits"):
        np.testing.assert_array_equal(actual[name], expected[name])
