# drama3-mlx

**Fish Audio Drama 3 text-to-speech on Apple Silicon, with MLX.** Built with Fish Audio.

An unofficial port of [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)
(35B parameters, 3B active per token) from Fish Speech v3.0.0-beta's PyTorch code to
[MLX](https://github.com/ml-explore/mlx). Fish's own runtime needs an NVIDIA GPU with
48 GB or more; this one runs on a Mac with enough unified memory. It is not made,
sponsored or endorsed by Fish Audio.

- Loads Fish's release as published (block-FP8 experts, 41.5 GB) and, by default,
  re-quantizes the experts to 8-bit as it loads: same memory, three times faster,
  as close to the full-precision model as Fish's own bf16 runtime. 4-bit fits 64 GB Macs.
- Everything the open-source runtime does: `[direction]` tags, paired tags such as
  `<whisper>…</whisper>`, `<|speaker:0|>`…`<|speaker:15|>` multi-character scenes,
  zero-shot voice cloning from a clip and its transcript, long text chunked like
  Fish's server, and editing as continuation (`previous_takes`).
- Matches Fish's PyTorch reference layer by layer on the real weights (see
  [Verification](#verification)).

## Requirements

- Apple Silicon Mac, macOS 14 or later, Python 3.10+
- Unified memory, measured on an M5 Max while generating:

  | Experts (`experts=`) | Resident | Peak |
  |---|---|---|
  | `q8` (default) | 42 GB | 48–55 GB |
  | `q4` | 28 GB | 32 GB |
  | `fp8` (as released) | 42 GB | 47 GB |

## Install

```bash
pip install "git+https://github.com/Sogni-AI/drama3-mlx@v0.1.1"
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

## Use

```bash
# Speech
drama3-mlx generate --model checkpoints/drama-3 \
  --text "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair." \
  --out fine.wav

# A cloned voice: a clip and its exact transcript
drama3-mlx generate --model checkpoints/drama-3 --text "Hello there." \
  --ref-audio voice.wav --ref-text "What the clip says, word for word." --out hello.wav

# 4-bit experts for a 64 GB Mac (re-quantized at load; nothing written to disk)
drama3-mlx generate --model checkpoints/drama-3 --experts q4 --text "Hello." --out hello.wav

# Or keep a smaller copy on disk (25 GB at 4-bit) that loads without re-quantizing
drama3-mlx convert checkpoints/drama-3 checkpoints/drama-3-mlx-4bit --bits 4
```

```python
import soundfile as sf
from drama3_mlx import Drama3

tts = Drama3.load("checkpoints/drama-3")  # experts="q8" by default
audio, rate = sf.read("voice.wav", dtype="float32", always_2d=True)
voice = tts.reference(audio.mean(axis=1), rate, "What the clip says, word for word.")

line = "<|speaker:0|> [nervous] Did you hear that? <|speaker:1|> <whisper>Stay quiet.</whisper>"
sf.write("scene.wav", tts.synthesize(line, references=[voice]), tts.sample_rate)

# Editing as continuation: re-read a line with one word changed
take = next(tts.generate("It's due on Wednesday.", references=[voice]))
edit = next(tts.generate("It's due on Thursday.", references=[voice],
                         previous_takes=[(take.text, take.codes)]))
```

Sampling defaults are Fish's (temperature 0.7, top-p 0.7, top-k 30), with the same
repetition-aware resampling of the main token.

## Speed

On an M5 Max (128 GB), cloning a 3.9 s reference, end to end including the reference
prefill and the codec:

| Experts | Per frame (21.5 frames = 1 s of audio) | Real-time factor |
|---|---|---|
| `q8` | 25 ms | 0.54 (1.85× faster than real time) |
| `q4` | 27 ms | 0.59 |
| `fp8` | 133 ms | 2.9 (dequantizes the selected experts every step) |

Loading takes 5–10 s from a warm file cache. The experts are re-quantized layer by
layer as they stream in, so loading never holds the whole FP8 release: memory peaks
near the final size (28 GB for `q4`, measured including a 264-row generation). A request has about 1 s of fixed cost
(the reference prefill) on top of the per-frame time.

## Sampling: stalls at Fish's SDK defaults

At Fish's SDK defaults (temperature 0.7, top-p 0.7, top-k 30), which this package keeps
for fidelity, Drama 3 sometimes stalls: it goes silent and drops the rest of a line. On
the model card's own A1 example ("[sighs] Fine. If we miss it, it's on you. [laughs]
Well, maybe that's fair.", cloning its reference clip), 14 of 24 takes were complete;
Fish's PyTorch sampler failed both takes we ran. At temperature 1.0 / top-p 0.9 (the
model card's serving settings) 20 of 24 were complete, and a plain untagged line 16 of
16. Every incomplete take contained a silence of 2 s or more, so a caller can catch
them and sample the chunk again:

```python
import numpy as np

def no_stall(segment, window=4410):  # 0.1 s at 44.1 kHz
    frames = segment.audio[: len(segment.audio) // window * window].reshape(-1, window)
    quiet = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9) < -50
    run = longest = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    return longest < 20  # under 2 s

tts.synthesize(text, references=[voice], temperature=1.0, top_p=0.9,
               accept=no_stall, attempts=3)
```

With those settings, 16 of 16 takes of A1 were complete.

## How it works

| Fish Speech v3 (PyTorch) | drama3-mlx |
|---|---|
| Qwen3.5-MoE backbone (Gated DeltaNet + gated attention, 256 experts) | mlx-lm's Qwen3.5 decoder, with Fish's zero-centred fp32 norms and fp32 routing |
| Block-FP8 experts (Triton W8A8 on SM89+, dequantized elsewhere) | `mx.from_fp8` dequantization of the selected experts, or MLX affine 8/4-bit |
| Codebook-MTP tower: 4 shared + 9 residual layers over time | Ported module by module |
| 44.1 kHz, 10-codebook DAC codec (`codec.pth`) | mlx-audio's `fish_s1_dac`, read from `codec.pth` without torch |
| Prompt template, chunking, sampling, generation loop | Ported line for line |

## Verification

`tests/` compares the port against Fish's PyTorch code
([fish-speech v3.0.0-beta](https://github.com/fishaudio/fish-speech/releases/tag/v3.0.0-beta)):

- **Tiny-model parity:** a randomly initialised checkpoint in the exact release
  format (block-FP8 experts, same tensor names) runs through both implementations, in
  fp32 on the CPU, through chunked prefill and single-row decode steps. Every backbone
  layer, the hidden states and the main-token and codebook logits agree within
  2e-4 of the largest value (measured: under 7e-5).
- **Real-model parity:** Fish's published A1 example (its reference clip and text, and
  Fish's own output audio, encoded) teacher-forced through both implementations on the
  41.5 GB release, fp32 on the CPU, 264 rows. Every one of the 40 layers agrees within
  2.2e-6 per row, the main-token logits within 7.7e-7 and the codebook logits within
  7.7e-5, the same level at which Fish's code agrees with itself under a different
  prefill chunking (4.2e-5); greedy choices agree on all 264 rows and 2,367 codes.
  In bf16 on the GPU the port is as close to fp32 as Fish's own bf16 run (main-token
  greedy agreement 90.5% vs 89.9%, codebook 92.0% vs 91.6%).
- **Generation loop:** with both samplers replaced by the same deterministic rule, the
  MLX loop and Fish's `generate_drama3` produce identical frames, including the
  repetition-aware resampling.
- Prompt rows on the real tokenizer, text chunking, the sampler, `codec.pth` reading
  (bit-exact against `torch.load`) and the converter.

The reference-backed tests run when these are set; otherwise they skip:

```bash
pip install -e '.[test]'
DRAMA3_REF_PYTHON=/path/to/python-with-torch \
DRAMA3_FISH_SPEECH=/path/to/fish-speech-v3.0.0-beta \
DRAMA3_CHECKPOINT=checkpoints/drama-3 \
pytest
```

On M5-generation GPUs MLX's fp32 matrix multiply is TF32-like, so the exact-parity
tests run MLX on the CPU.

## License

The Drama 3 weights, the codec and Fish Speech are Fish Audio Materials under the
[Fish Audio Community License, Version 1.0](LICENSE), and this port is a Derivative
Work under the same license (see [NOTICE](NOTICE)). In short: research and
non-commercial use are free; commercial use is licensed only while your annual
revenue (with affiliates) is under USD 500,000; offering Drama 3 as a hosted
text-to-speech or voice-cloning service needs a separate license from Fish Audio
(business@fish.audio) whatever your revenue. Do not clone a real person's voice
without their consent. Read the LICENSE for the actual terms.

The MLX, mlx-lm and mlx-audio dependencies are MIT-licensed.
