"""Shared fixtures.

Tests that compare against Fish's PyTorch reference run only when these are set:

    DRAMA3_REF_PYTHON   a Python with torch, safetensors, tokenizers, loguru, tqdm
    DRAMA3_FISH_SPEECH  a fish-speech v3.0.0-beta checkout
    DRAMA3_CHECKPOINT   fishaudio/drama-3 (tokenizer.json and codec.pth are used)
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).parent


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return Path(value)


@pytest.fixture(scope="session")
def ref_python() -> Path:
    return env_path("DRAMA3_REF_PYTHON")


@pytest.fixture(scope="session")
def fish_speech() -> Path:
    return env_path("DRAMA3_FISH_SPEECH")


@pytest.fixture(scope="session")
def checkpoint() -> Path:
    return env_path("DRAMA3_CHECKPOINT")


@pytest.fixture(scope="session")
def run_reference(ref_python, fish_speech):
    """Run a Python snippet against the reference; it prints one JSON value."""

    def run(code: str, *args: str):
        result = subprocess.run(
            [str(ref_python), "-I", "-c", code, str(fish_speech), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    return run


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, ref_python) -> Path:
    out = tmp_path_factory.mktemp("tiny") / "checkpoint"
    subprocess.run(
        [str(ref_python), "-I", str(TESTS / "reference" / "make_tiny_checkpoint.py"), str(out)],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def tiny_release_tokens(tmp_path_factory, ref_python, checkpoint) -> Path:
    """A tiny checkpoint with the release's tokenizer, token layout and codec."""
    out = tmp_path_factory.mktemp("tiny-release-tokens") / "checkpoint"
    subprocess.run(
        [
            str(ref_python), "-I", str(TESTS / "reference" / "make_tiny_checkpoint.py"),
            str(out), "--release-tokens", str(checkpoint),
        ],
        check=True,
        capture_output=True,
    )
    return out
