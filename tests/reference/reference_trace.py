"""Record a teacher-forced trace of the PyTorch reference (fish-speech v3 fish_omni).

Rows [N, 1 + num_codebooks] are fed the way generate_drama3 feeds them: the first
--prompt-rows rows as chunked prefill, every later row as a single-row decode step,
and the codebook tower of row p is forced with the codes of row p + 1 instead of
sampling them. The trace holds what the sampler would see at every step:

    hidden       [N, H]            backbone final-norm hidden states
    main_logits  [N, 1 + band]     lm_head logits over eos + the semantic band
    tower_logits [N - 1, C - 1, V] codebook-head logits, head i predicting code i + 1

Needs torch and a fish-speech v3.0.0-beta checkout (not MLX):

    python tests/reference/reference_trace.py --fish-speech PATH --checkpoint DIR \
        --rows rows.npy --prompt-rows R --out trace.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fish-speech", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--prompt-rows", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument(
        "--keep-fp8",
        action="store_true",
        help="keep experts in FP8 and dequantize per forward (41 GB instead of 70 GB)",
    )
    parser.add_argument(
        "--layers", action="store_true", help="also record every backbone layer output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.fish_speech.resolve()))
    import torch

    import fish_speech.models.text2semantic.fish_omni.model as omni_model
    from fish_speech.models.text2semantic.fish_omni.generate import (
        ensure_drama3_caches,
    )

    if args.keep_fp8:
        omni_model.fp8_gemm_supported = lambda devices: True
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    started = time.perf_counter()
    model = omni_model.FishOmniModel.from_pretrained(
        args.checkpoint, devices=["cpu"], tower_device="cpu", dtype=dtype
    )
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)
    backbone, tower, config = model.backbone, model.audio_decoder, model.config

    rows = torch.from_numpy(np.load(args.rows)).long()
    total, prompt_rows = len(rows), args.prompt_rows
    assert 0 < prompt_rows <= total
    ensure_drama3_caches(model, total + 1)
    positions = torch.arange(total)
    candidates = torch.cat(
        (
            torch.tensor([config.eos_token_id]),
            torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1),
        )
    ).sort().values

    layer_outputs: list[list[torch.Tensor]] = [[] for _ in backbone.model.layers]
    if args.layers:
        for index, layer in enumerate(backbone.model.layers):
            layer.register_forward_hook(
                lambda module, inputs, output, index=index: layer_outputs[index].append(
                    output.float().clone()
                )
            )

    def forced_tower(hidden, start, end):
        """Head logits of rows start..end-1, each forced with the next row's codes."""
        current, following = rows[start:end], rows[start + 1 : end + 1]
        semantic = tower.is_semantic(following[:, 0])
        out = tower.run_shared(
            hidden, following[:, 1], semantic, positions[start:end], current
        )
        logits = []
        for index, (layer, head) in enumerate(
            zip(tower.residual_layers, tower.codebook_heads)
        ):
            out = layer(out, following[:, index + 1], semantic, positions[start:end])
            logits.append(head(out).float())
        return torch.stack(logits, dim=1)

    hidden_rows, tower_rows = [], []
    with torch.inference_mode():
        started = time.perf_counter()
        for start in range(0, prompt_rows, args.prefill_chunk):
            end = min(start + args.prefill_chunk, prompt_rows)
            chunk = rows[start:end]
            embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
            hidden = backbone(
                inputs_embeds=embeddings[None], input_pos=positions[start:end]
            )[0]
            hidden_rows.append(hidden)
            # As in generate_drama3, the last prompt row waits for the first step.
            paired_end = min(end, prompt_rows - 1)
            if paired_end > start:
                tower_rows.append(
                    forced_tower(hidden[: paired_end - start], start, paired_end)
                )
        print(f"prefill {prompt_rows} rows in {time.perf_counter() - started:.1f}s")

        started = time.perf_counter()
        for position in range(prompt_rows - 1, total - 1):
            # Decode step at position: tower on this row, then the backbone on the next.
            tower_rows.append(
                forced_tower(hidden_rows[-1][-1:], position, position + 1)
            )
            row = rows[position + 1 : position + 2]
            embeddings = tower.embed_backbone(row, backbone.embed(row[:, 0]))
            hidden_rows.append(
                backbone(
                    inputs_embeds=embeddings[None],
                    input_pos=positions[position + 1 : position + 2],
                )[0]
            )
        print(
            f"decode {total - prompt_rows} rows in {time.perf_counter() - started:.1f}s"
        )

        hidden = torch.cat(hidden_rows)
        main_logits = backbone.compute_logits(hidden)[:, candidates].float()

    trace = {
        "hidden": hidden.float().numpy(),
        "main_logits": main_logits.numpy(),
        "tower_logits": torch.cat(tower_rows).numpy(),
        "candidates": candidates.numpy(),
    }
    if args.layers:
        for index, outputs in enumerate(layer_outputs):
            trace[f"layer_{index}"] = torch.cat(outputs, dim=1)[0].numpy()
    np.savez(args.out, **trace)
    print(f"wrote {args.out}: " + ", ".join(f"{k} {v.shape}" for k, v in trace.items()))


if __name__ == "__main__":
    main()
