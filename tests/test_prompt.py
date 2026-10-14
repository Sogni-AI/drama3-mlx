"""Prompt rows and text chunks against fish-speech v3.0.0-beta, on the real tokenizer."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer

from drama3_mlx.chunking import split_text
from drama3_mlx.config import Drama3Config
from drama3_mlx.prompt import InvalidInputError, build_prompt

REFERENCE_PROMPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
from tokenizers import Tokenizer
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
checkpoint, case = sys.argv[2], json.loads(sys.argv[3])
config = FishOmniConfig.from_dict(json.load(open(checkpoint + "/config.json")))
tokenizer = Tokenizer.from_file(checkpoint + "/tokenizer.json")
rows = build_drama3_prompt(
    case["text"], tokenizer, config, case["ref_texts"],
    [torch.tensor(c) for c in case["ref_codes"]],
    [(t, torch.tensor(c)) for t, c in case["history"]],
)
print(json.dumps(rows.tolist()))
"""

REFERENCE_CHUNKS = """
import json, sys
sys.path.insert(0, sys.argv[1])
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
print(json.dumps([split_drama3_text(text) for text in json.loads(sys.argv[2])]))
"""

CODES = np.arange(30).reshape(10, 3) % 4096
CASES = [
    {"text": "Hello there.", "ref_texts": [], "ref_codes": [], "history": []},
    {
        "text": "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.",
        "ref_texts": ["  A reference transcript with leading space."],
        "ref_codes": [CODES.tolist()],
        "history": [],
    },
    {
        "text": "<|speaker:1|> Where were you? <|speaker:0|> <whisper>Out.</whisper>",
        "ref_texts": ["<|speaker:0|> First voice.", "<|speaker:1|> Second voice."],
        "ref_codes": [CODES.tolist(), (CODES[:, :2] + 7).tolist()],
        "history": [["Earlier chunk.", (CODES + 1).tolist()]],
    },
    {"text": "你好，欢迎使用语音合成服务。Mixed English too.", "ref_texts": [], "ref_codes": [], "history": []},
]
CHUNK_TEXTS = [
    "Short text.",
    " ".join(["This is a sentence that keeps going, on and on."] * 20),
    "<|speaker:0|> " + "Alpha speaks at length. " * 15 + "<|speaker:1|> Beta replies. "
    "<|speaker:2|> Gamma. <|speaker:3|> Delta joins. <|speaker:0|> [laughing] Back to alpha.",
    "Pi is 3.14159 and e is 2.71828. " * 12 + "<whisper>" + "secret " * 60 + "</whisper>",
    "他停顿了一下，低声说道：其实，我早就知道这件事了，只是一直没说出口。" * 6,
]


@pytest.fixture(scope="module")
def tokenizer_config(checkpoint):
    config = Drama3Config.from_dict(json.loads((checkpoint / "config.json").read_text()))
    return Tokenizer.from_file(str(checkpoint / "tokenizer.json")), config


@pytest.mark.parametrize("case", CASES)
def test_prompt_matches_reference(case, checkpoint, tokenizer_config, run_reference):
    tokenizer, config = tokenizer_config
    rows = build_prompt(
        case["text"],
        tokenizer,
        config,
        case["ref_texts"],
        [np.array(c) for c in case["ref_codes"]],
        [(t, np.array(c)) for t, c in case["history"]],
    )
    expected = run_reference(REFERENCE_PROMPT, str(checkpoint), json.dumps(case))
    assert rows.tolist() == expected


def test_chunks_match_reference(run_reference):
    expected = run_reference(REFERENCE_CHUNKS, json.dumps(CHUNK_TEXTS))
    assert [split_text(text) for text in CHUNK_TEXTS] == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "Hi <|im_end|>", "<|speaker:16|> too high", "<|speaker:01|> padded", "<|speaker:x|>"],
)
def test_rejects_invalid_text(text, tokenizer_config):
    tokenizer, config = tokenizer_config
    with pytest.raises(InvalidInputError):
        build_prompt(text, tokenizer, config)
