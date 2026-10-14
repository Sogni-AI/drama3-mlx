"""The MLX counterpart of tests/reference/reference_trace.py.

Feeds rows exactly as the reference trace does (chunked prefill of the prompt rows,
then single-row decode steps, the tower forced with the next row's codes) and
returns the same arrays, so the two can be compared element by element.

    python tests/mlx_trace.py --checkpoint DIR --rows rows.npy --prompt-rows R --out trace.npz
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from drama3_mlx.generate import PREFILL_CHUNK_SIZE, run_backbone
from drama3_mlx.loading import load_model


def backbone_rows(model, frames, caches, layer_outputs=None) -> mx.array:
    """run_backbone, optionally recording every layer's output (residual stream)."""
    if layer_outputs is None:
        return run_backbone(model, frames, caches)
    text_model, tower = model.backbone.model, model.tower
    hidden = tower.embed_backbone(frames, text_model.embed_tokens(frames[:, 0]))[None]
    fa_mask = create_attention_mask(hidden, caches[text_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden, caches[text_model.ssm_idx])
    for index, (layer, cache) in enumerate(zip(text_model.layers, caches)):
        hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache)
        layer_outputs[index].append(hidden[0].astype(mx.float32))
    return text_model.norm(hidden)[0]


def mlx_trace(model, rows: np.ndarray, prompt_rows: int, *, layers: bool = False,
              prefill_chunk: int = PREFILL_CHUNK_SIZE) -> dict[str, np.ndarray]:
    config, backbone, tower = model.config, model.backbone, model.tower
    total = len(rows)
    frames = mx.array(rows.astype(np.int32))
    candidates = mx.array(
        sorted([config.eos_token_id, *range(config.semantic_start_token_id,
                                            config.semantic_end_token_id + 1)])
    )
    caches, tower_caches = backbone.make_cache(), tower.make_caches()

    layer_outputs: list[list[mx.array]] = [[] for _ in backbone.model.layers]

    hidden_rows, tower_rows = [], []
    for start in range(0, prompt_rows, prefill_chunk):
        end = min(start + prefill_chunk, prompt_rows)
        hidden = backbone_rows(model, frames[start:end], caches, layer_outputs if layers else None)
        hidden_rows.append(hidden)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower_rows.append(
                tower.forced(
                    hidden[: paired_end - start],
                    frames[start:paired_end],
                    frames[start + 1 : paired_end + 1],
                    tower_caches,
                )
            )
        mx.eval(hidden_rows, tower_rows)

    for position in range(prompt_rows - 1, total - 1):
        tower_rows.append(
            tower.forced(
                hidden_rows[-1][-1:],
                frames[position : position + 1],
                frames[position + 1 : position + 2],
                tower_caches,
            )
        )
        hidden_rows.append(
            backbone_rows(model, frames[position + 1 : position + 2], caches,
                          layer_outputs if layers else None)
        )
        mx.eval(hidden_rows[-1], tower_rows[-1])

    hidden = mx.concatenate(hidden_rows)
    main_logits = backbone.lm_head(hidden)[:, candidates]
    trace = {
        "hidden": hidden,
        "main_logits": main_logits,
        "tower_logits": mx.concatenate(tower_rows),
        "candidates": candidates,
    }
    if layers:
        for index, outputs in enumerate(layer_outputs):
            trace[f"layer_{index}"] = mx.concatenate(outputs)
    return {name: np.array(value.astype(mx.float32) if value.dtype != mx.int32 else value)
            for name, value in trace.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--prompt-rows", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--layers", action="store_true")
    parser.add_argument("--experts", default="fp8", help="fp8 (exact, default), q8, q4, bf16")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="run on the CPU: exact fp32 matmuls (the GPU's fp32 GEMM is TF32-like on M5)",
    )
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    # Freed dequantized experts otherwise stay in MLX's buffer cache (92 GB peak).
    mx.set_cache_limit(2 * 1024**3)
    started = time.perf_counter()
    model = load_model(args.checkpoint, dtype=getattr(mx, args.dtype), experts=args.experts)
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)
    started = time.perf_counter()
    trace = mlx_trace(model, np.load(args.rows), args.prompt_rows, layers=args.layers)
    print(f"traced in {time.perf_counter() - started:.1f}s")
    np.savez(args.out, **trace)


if __name__ == "__main__":
    main()
