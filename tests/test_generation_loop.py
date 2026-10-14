"""The generation loop against fish-speech's generate_drama3, made deterministic.

Teacher-forced parity covers the logits; this covers what sampling adds: the loop
wiring, the repetition-aware resampling (RAS) window and substitution, and the
stop. Both samplers are replaced by the same rule, argmax for an ordinary draw and
the second-ranked token for the RAS retry (the only draw at temperature 1.0), so the
two loops must produce identical frames on the tiny checkpoint, in fp32 on the CPU.
"""

import json
import subprocess

import mlx.core as mx
import numpy as np

import drama3_mlx.tower as tower_module
from drama3_mlx.generate import generate_frames
from drama3_mlx.loading import load_model
from drama3_mlx.sampling import RAS_TEMPERATURE

from .test_parity_tiny import PROMPT_ROWS, tiny_rows

MAX_NEW_TOKENS = 60

REFERENCE = """
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import torch
import fish_speech.models.text2semantic.fish_omni.model as om
import fish_speech.models.text2semantic.fish_omni.tower as tower
from fish_speech.models.text2semantic.fish_omni.generate import generate_drama3

def deterministic(logits, temperature, top_k, top_p, **_):
    ranked = torch.argsort(logits.float(), dim=-1, descending=True)
    return ranked[:, 1] if float(temperature) == float(sys.argv[4]) else ranked[:, 0]

tower.sample_top_k_top_p = deterministic
model = om.FishOmniModel.from_pretrained(sys.argv[2], devices=["cpu"], tower_device="cpu", dtype=torch.float32)
prompt = torch.from_numpy(np.load(sys.argv[3])).long()
frames, finish = generate_drama3(model, prompt, max_new_tokens=int(sys.argv[5]))
print(json.dumps({"frames": frames.tolist(), "finish": finish}))
"""


def deterministic(logits, temperature, top_k, top_p):
    ranked = mx.argsort(-logits.astype(mx.float32), axis=-1)
    return ranked[:, 1] if temperature == RAS_TEMPERATURE else ranked[:, 0]


def test_loop_matches_reference(tmp_path, tiny_checkpoint, ref_python, fish_speech, monkeypatch):
    prompt = tiny_rows()[:PROMPT_ROWS]
    np.save(tmp_path / "prompt.npy", prompt)
    result = subprocess.run(
        [
            str(ref_python), "-I", "-c", REFERENCE, str(fish_speech), str(tiny_checkpoint),
            str(tmp_path / "prompt.npy"), str(RAS_TEMPERATURE), str(MAX_NEW_TOKENS),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(result.stdout.strip().splitlines()[-1])

    monkeypatch.setattr(tower_module, "sample_top_k_top_p", deterministic)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = load_model(tiny_checkpoint, dtype=mx.float32, experts="fp8")
        frames, finish = generate_frames(model, prompt, max_new_tokens=MAX_NEW_TOKENS)
        # The same loop with the retry equal to the first draw, i.e. no RAS.
        monkeypatch.setattr(
            tower_module, "sample_top_k_top_p", lambda logits, *_: deterministic(logits, 0, 0, 0)
        )
        without_ras, _ = generate_frames(model, prompt, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        mx.set_default_device(previous)

    assert finish == expected["finish"]
    assert frames.tolist() == expected["frames"]
    # Only meaningful if RAS substituted tokens along the way.
    assert without_ras.tolist() != frames.tolist()
