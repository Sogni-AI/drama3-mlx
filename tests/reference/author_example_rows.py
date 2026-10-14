"""Rows for a real-checkpoint parity trace from one of Fish's published examples.

The prompt is the example's text cloning the example's reference clip; the frames
after it are Fish's own published output, encoded with the codec. Both traces then
score exactly the audio the author's server produced.

    python tests/reference/author_example_rows.py --checkpoint DRAMA3_DIR \
        --text "..." --ref-audio en0.wav --ref-text "..." --output A1_point_tags.wav \
        --frames 96 --out rows.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from drama3_mlx.codec import FishCodec
from drama3_mlx.config import Drama3Config
from drama3_mlx.prompt import build_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--output", type=Path, required=True, help="the author's output audio")
    parser.add_argument("--frames", type=int, default=96, help="output frames to append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = Drama3Config.from_dict(json.loads((args.checkpoint / "config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    codec = FishCodec(args.checkpoint / "codec.pth")

    def encode(path: Path) -> np.ndarray:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return codec.encode(audio.mean(axis=1), sample_rate)

    prompt = build_prompt(args.text, tokenizer, config, [args.ref_text], [encode(args.ref_audio)])
    output = encode(args.output)[:, : args.frames].astype(np.int64)
    frames = np.concatenate([output[:1] + config.semantic_start_token_id, output]).T
    rows = np.concatenate([prompt, frames])
    np.save(args.out, rows)
    print(json.dumps({"prompt_rows": len(prompt), "rows": len(rows)}))


if __name__ == "__main__":
    main()
