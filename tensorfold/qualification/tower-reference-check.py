#!/usr/bin/env python3
"""Actual production tower vs pinned unfused AudioDecoder with its DecoderAttention protocol.

Attention uses PyTorch SDPA's math backend, not SGLang's paged backend. This is a
teacher-forced component check, not whole-model or fused decode qualification.
Run only inside run-bounded-oracle.py's bounded, full-lifetime GPU guard.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROWS = (1, 7, 128)
THRESHOLDS = {"minimum_cosine": 0.999, "maximum_relative_l2": 0.03,
              "head_top1_equal": True, "minimum_head_top30_overlap": 0.9,
              "maximum_head_kl": 0.01}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bf16_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2").reshape(shape)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("shape mismatch or non-finite tower output")
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    error = np.linalg.norm(a - b)
    cosine = float(np.dot(a, b) / (an * bn)) if an and bn else float(an == bn)
    relative = float(error / bn) if bn else (0.0 if error == 0 else float("inf"))
    return {"max_abs": float(np.max(np.abs(a - b))),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "relative_l2": relative, "cosine": cosine,
            "equal_elements": int(np.count_nonzero(actual == expected)),
            "elements": a.size,
            "pass": cosine >= THRESHOLDS["minimum_cosine"] and relative <= THRESHOLDS["maximum_relative_l2"]}


def head_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    result = metrics(actual, expected)
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    a_order, b_order = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
    a_lp = a - a.max(); a_lp -= np.log(np.exp(a_lp).sum())
    b_lp = b - b.max(); b_lp -= np.log(np.exp(b_lp).sum())
    overlap = len(set(a_order[:30]) & set(b_order[:30])) / 30
    kl = float(np.sum(np.exp(b_lp) * (b_lp - a_lp)))
    result.update(top1_equal=bool(a_order[0] == b_order[0]),
                  native_top1=int(a_order[0]), reference_top1=int(b_order[0]),
                  top30_overlap=overlap, kl_reference_to_native=kl)
    result["pass"] = result["pass"] and result["top1_equal"] and overlap >= THRESHOLDS["minimum_head_top30_overlap"] and kl <= THRESHOLDS["maximum_head_kl"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="sglang-omni checkout at its pinned revision")
    parser.add_argument("--engine", type=Path, required=True, help="patched TensorFold tree with its oracle tools built")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--input-hidden", type=Path, help="External row-major BF16 hidden states; first 128 rows")
    parser.add_argument("--input-frames", type=Path, help="External u32le frame rows; first 129 rows")
    args = parser.parse_args()
    args.native = args.engine / "zig-out/oracle/bin/tf-drama-tower-oracle"
    if (args.input_hidden is None) != (args.input_frames is None):
        parser.error("--input-hidden and --input-frames must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config_data = json.loads((args.checkpoint / "config.json").read_text())
    if args.input_hidden is None:
        generator = torch.Generator(device="cpu").manual_seed(20261014)
        hidden = torch.randn((128, 2048), generator=generator, dtype=torch.float32).to(torch.bfloat16)
        frames = torch.randint(0, 4096, (129, 11), generator=generator, dtype=torch.int64)
        frames[:, 0] = frames[:, 1] + config_data["semantic_start_token_id"]
        frames[::4, 0] = 42  # Nonsemantic text rows, interleaved with semantic audio rows.
        fixture = {"fixture_kind": "synthetic random", "fixture_seed": 20261014}
    else:
        raw_hidden = np.fromfile(args.input_hidden, dtype="<u2")
        raw_frames = np.fromfile(args.input_frames, dtype="<u4")
        if raw_hidden.size < 128 * 2048 or raw_hidden.size % 2048:
            raise ValueError("external hidden data must contain at least 128 complete rows of width 2048")
        if raw_frames.size < 129 * 11 or raw_frames.size % 11:
            raise ValueError("external frame data must contain at least 129 complete rows of width 11")
        hidden = torch.from_numpy(raw_hidden.reshape(-1, 2048)[:128].copy()).view(torch.bfloat16)
        frames = torch.from_numpy(raw_frames.reshape(-1, 11)[:129].astype(np.int64))
        if not torch.isfinite(hidden.float()).all() or torch.any(frames > np.iinfo(np.int32).max):
            raise ValueError("non-finite hidden data or frame value outside native signed range")
        fixture = {"fixture_kind": "external tensors", "fixture_seed": None,
                   "source_hidden_sha256": sha256(args.input_hidden),
                   "source_frames_sha256": sha256(args.input_frames)}
    hidden.view(torch.uint16).numpy().astype("<u2").tofile(args.output / "hidden.bf16")
    frames.numpy().astype("<u4").tofile(args.output / "frames.u32le")
    result = {**fixture, "thresholds": THRESHOLDS,
              "hidden_sha256": sha256(args.output / "hidden.bf16"),
              "frames_sha256": sha256(args.output / "frames.u32le"),
              "reference_regime": "Pinned unfused AudioDecoder.prefill; PyTorch math SDPA through the official DecoderAttention protocol; no sampling",
              "scope": "all13 tower layers, final hidden, layer KV rows and final codebook9 head; not fused decode, sampler or whole-model parity"}
    (args.output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"prepared": True, "cuda_initialized": torch.cuda.is_initialized(), **result}))
        return 0
    result["native_binary_sha256"] = sha256(args.native)
    with (args.output / "native.log").open("w") as log:
        subprocess.run([str(args.native), str(args.checkpoint), str(args.output), str(args.output)], stdout=log, stderr=subprocess.STDOUT, check=True)
    sys.path.insert(0, str(args.reference))
    from safetensors import safe_open
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
    from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
    from sglang_omni.models.fishaudio_drama3.weights import load_audio_decoder_weights
    torch.cuda.set_per_process_memory_fraction((8 << 30) / torch.cuda.get_device_properties(0).total_memory)
    config = FishOmniConfig.from_dict(config_data)
    config.text_config.max_position_embeddings = 128
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            decoder = AudioDecoder(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)

    def tensors():
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        names_by_file = {}
        for name, filename in index.items():
            if name.startswith("audio_decoder"):
                names_by_file.setdefault(filename, []).append(name)
        for filename, names in sorted(names_by_file.items()):
            with safe_open(str(args.checkpoint / filename), framework="pt", device="cpu") as reader:
                for name in sorted(names):
                    yield name, reader.get_tensor(name)

    load_audio_decoder_weights(decoder, tensors())
    assert decoder.fused_layers is None
    input_hidden, input_frames = hidden.cuda(), frames.cuda()
    captured = {}
    handle = decoder.residual_layers[-1].register_forward_hook(lambda module, inputs, output: captured.update(hidden=output.detach().clone()))

    def reference_chunked(n):
        """Same reference layers, one row at a time, with growing per-layer KV."""
        cache = {}
        outputs = []
        for position in range(n):
            def attention(layer_index, query, key, value):
                if layer_index in cache:
                    old_key, old_value = cache[layer_index]
                    key = torch.cat((old_key, key), dim=0)
                    value = torch.cat((old_value, value), dim=0)
                cache[layer_index] = (key, value)
                q = query.view(1, 16, 256).transpose(0, 1).unsqueeze(0)
                k = key.view(position + 1, 2, 256).transpose(0, 1).unsqueeze(0)
                v = value.view(position + 1, 2, 256).transpose(0, 1).unsqueeze(0)
                # Every cached row precedes this query: a top-left causal mask
                # would incorrectly mask all but the first cached key.
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
                return out.squeeze(0).transpose(0, 1).reshape(1, 4096)
            decoder.prefill(input_hidden[position:position + 1].clone(),
                            torch.tensor([position], device="cuda"),
                            input_frames[position:position + 1],
                            input_frames[position + 1:position + 2], attention)
            outputs.append(captured["hidden"])
        return torch.cat(outputs, dim=0)

    cases = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for n in ROWS:
            kv = {}

            def attention(layer_index, query, key, value):
                kv[layer_index] = (key.detach().float().cpu().numpy(), value.detach().float().cpu().numpy())
                q = query.view(n, 16, 256).transpose(0, 1).unsqueeze(0)
                k = key.view(n, 2, 256).transpose(0, 1).unsqueeze(0)
                v = value.view(n, 2, 256).transpose(0, 1).unsqueeze(0)
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                return out.squeeze(0).transpose(0, 1).reshape(n, 4096)

            decoder.prefill(input_hidden[:n].clone(), torch.arange(n, device="cuda"), input_frames[:n], input_frames[1:n + 1], attention)
            final_hidden = captured["hidden"]
            ref_hidden = final_hidden.float().cpu().numpy()
            ref_logits = decoder.codebook_heads[-1](final_hidden[-1:]).float().cpu().numpy().reshape(-1)
            np.save(args.output / f"reference-{n}-hidden.npy", ref_hidden)
            np.save(args.output / f"reference-{n}-head9.npy", ref_logits)
            native_hidden = bf16_array(args.output / f"native-{n}-hidden.bf16", (n, 2048))
            native_chunked = bf16_array(args.output / f"native-{n}-chunked-hidden.bf16", (n, 2048))
            native_logits = bf16_array(args.output / f"native-{n}-head9.bf16", (4096,))
            layer_metrics = []
            for layer_index in range(13):
                row = {"layer": layer_index}
                for j, kind in enumerate(("keys", "values")):
                    native = bf16_array(args.output / f"native-{n}-layer-{layer_index}-{kind}.bf16", (n, 512))
                    reference = kv[layer_index][j]
                    np.save(args.output / f"reference-{n}-layer-{layer_index}-{kind}.npy", reference)
                    row[kind] = metrics(native, reference)
                layer_metrics.append(row)
            case = {"rows": n, "native_reset_bit_exact": True,
                    "hidden": metrics(native_hidden, ref_hidden),
                    "head9": head_metrics(native_logits, ref_logits),
                    "native_chunk_vs_batch": metrics(native_chunked, native_hidden),
                    "layers": layer_metrics}
            case["pass"] = case["hidden"]["pass"] and case["head9"]["pass"] and case["native_chunk_vs_batch"]["pass"] and all(row[kind]["pass"] for row in layer_metrics for kind in ("keys", "values"))
            cases.append(case)
            # A self-control distinguishes batch-shape sensitivity in this
            # random fixture from a discrepancy unique to the native path.
            chunked_hidden = reference_chunked(n)
            chunked_array = chunked_hidden.float().cpu().numpy()
            chunked_logits = decoder.codebook_heads[-1](chunked_hidden[-1:]).float().cpu().numpy().reshape(-1)
            np.save(args.output / f"reference-{n}-chunked-hidden.npy", chunked_array)
            np.save(args.output / f"reference-{n}-chunked-head9.npy", chunked_logits)
            case["reference_chunk_vs_batch"] = metrics(chunked_array, ref_hidden)
            case["reference_chunk_vs_batch_head9"] = head_metrics(chunked_logits, ref_logits)
            case["native_vs_reference_chunked"] = metrics(native_chunked, chunked_array)
            case["gates_unchanged"] = True
            print(json.dumps({k: v for k, v in case.items() if k != "layers"}), flush=True)
    handle.remove()
    result.update(cases=cases, passed=all(row["pass"] for row in cases),
                  torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                  sglang_version=importlib.metadata.version("sglang"),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                  reference_context=128, reference_fused_layers=False,
                  gaps=["Fused-vs-unfused reference difference unmeasured", "SGLang paged attention backend not exercised", "Reference whole-model hidden inputs not used"])
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
