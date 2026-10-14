"""drama3-mlx command line: generate speech, or convert Fish's release."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import convert as convert_module
from .model import Drama3
from .sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


def generate(args: argparse.Namespace) -> None:
    text = args.text if args.text is not None else Path(args.text_file).read_text()
    started = time.perf_counter()
    tts = Drama3.load(args.model, experts=args.experts)
    print(f"loaded {args.model} in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    references = []
    if args.ref_audio:
        if not args.ref_text:
            sys.exit("--ref-audio needs --ref-text, the exact transcript of the clip")
        audio, sample_rate = sf.read(args.ref_audio, dtype="float32", always_2d=True)
        references.append(tts.reference(audio.mean(axis=1), sample_rate, args.ref_text))
    pieces = []
    for segment in tts.generate(
        text,
        references=references,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        single_chunk=args.single_chunk,
        seed=args.seed,
    ):
        seconds = len(segment.audio) / tts.sample_rate
        print(
            f"chunk: {segment.codes.shape[1]} frames, {seconds:.2f}s audio in "
            f"{segment.seconds:.2f}s (RTF {segment.seconds / seconds:.2f}), "
            f"finish={segment.finish_reason}: {segment.text[:60]!r}",
            file=sys.stderr,
        )
        pieces.append(segment.audio)
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    sf.write(args.out, audio, tts.sample_rate)
    print(f"wrote {args.out} ({len(audio) / tts.sample_rate:.2f}s)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="drama3-mlx", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="text to speech")
    gen.add_argument("--model", required=True, help="fishaudio/drama-3 or a converted dir")
    source = gen.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    gen.add_argument("--out", required=True, help="output .wav")
    gen.add_argument(
        "--experts",
        choices=("q8", "q4", "fp8", "bf16"),
        default="q8",
        help="how the release's experts are held in memory (default q8)",
    )
    gen.add_argument("--ref-audio", help="reference clip of the voice to clone")
    gen.add_argument("--ref-text", help="exact transcript of --ref-audio")
    gen.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    gen.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    gen.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    gen.add_argument("--max-new-tokens", type=int, default=0, help="frames per chunk")
    gen.add_argument("--single-chunk", action="store_true", help="no long-text chunking")
    gen.add_argument("--seed", type=int)
    gen.set_defaults(handler=generate)

    conv = commands.add_parser("convert", help="convert Fish's release for MLX")
    conv.add_argument("src", type=Path)
    conv.add_argument("dst", type=Path)
    conv.add_argument("--bits", type=int, choices=(4, 8), help="expert bits (default: bf16)")
    conv.set_defaults(handler=lambda a: convert_module.convert(a.src, a.dst, bits=a.bits))

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
