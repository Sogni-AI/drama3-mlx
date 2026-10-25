#!/usr/bin/env python3
"""Reference-vs-reference control for the Drama 3 full-backbone gates.

The frozen fixture (--baseline, a backbone-reference-check.py output) is one solo EXTEND of the 182-row prompt
through the real Drama3EngineBuilder/OmniScheduler, captured by wrapping Drama3ForCausalLM.backbone_logits
(backbone-reference-check.py run_reference and capture_request). This script reuses that path with
max_running_requests=2. It re-runs the solo schedule as a determinism control (first and last), then co-batches
the prompt in one EXTEND with a companion request, a prefix of the same prompt, before or after it, and finally
with a second copy of itself. Every target view is scored against the frozen bytes with the native gate code,
imported unchanged from backbone-reference-check.py. The frozen native outputs are scored against each
view for context. No model arithmetic is reimplemented or patched. Run the full mode only inside the
bounded GPU guard with the confined launch from README.md, then bind the guard record with --finalize.
--plan-only checks every frozen input on the CPU and prints the plan; it never imports torch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import ipaddress
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HELPER_PATH = Path(__file__).resolve().parents[1] / "backbone-reference-check.py"
# Every home, temp and cache directory the runtime resolves, each under OUTPUT/cache/<name>.
CACHE_ENV = {"HOME": "home", "XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp", "TRITON_CACHE_DIR": "triton",
             "CUDA_CACHE_PATH": "cuda", "TORCHINDUCTOR_CACHE_DIR": "inductor",
             "FLASHINFER_WORKSPACE_BASE": "flashinfer", "HF_HOME": "hf"}
# --seed-cache copies the kernels an earlier guarded reference run compiled into its runtime-cache, without writing
# to that shared directory.
SEEDED_CACHES = ("triton", "cuda")
# Loaded as backbone-frozen-compare.py loads it; its sha256 must equal the one the frozen fixture records.
_spec = importlib.util.spec_from_file_location("frozen_backbone_comparison", HELPER_PATH)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
EvidenceError = helper.EvidenceError

MAX_RUNNING_REQUESTS = 2
# request_builders.py:24-25 refuses prompts under two rows, so the smallest companion is 2, not 1.
COMPANION_ROWS = (2, 7, 128)
CONTROL_PHASES = ("solo-first", "solo-last")
SCHEMA = "drama-backbone-reference-self/1"
SCOPE = ("The pinned SGLang-Omni reference against its own frozen solo-EXTEND output for the 182-row backbone fixture, "
         "when only the prefill call schedule changes (co-batched requests, row offset inside the call). The gate "
         "code and thresholds are the native comparison's. Not decode, tower, sampling or audio parity.")
SERVING_EVIDENCE = [
    "sglang-omni/sglang_omni/models/fishaudio_drama3/config.py:19 DEFAULT_MAX_RUNNING_REQUESTS = 16; config.py:62 "
    "and stages.py:150 make it the served default",
    "sglang-omni/docs/cookbook/fishaudio_drama3.md:197 'The defaults run at most 16 requests at a time on the "
    "engine'; :254 replays 'prefill batches of up to 2048 tokens'; :255 first-frame graphs 'for prefills of up to "
    "16 requests'",
    "sglang-omni/sglang_omni/models/fishaudio_drama3/sglang_model.py:217-237 forward_extend runs every request of a "
    "batch in one backbone_logits call (extend_start_loc/extend_seq_lens, one last row per request)",
    "Served defaults also replay the prefill as a breakable CUDA graph (engine_builder.py:138-153) and the first frame "
    "as a CUDA graph (config.py:27); this control runs eager like the frozen fixture "
    "(disable_cuda_graph=True). The pilot runs at concurrency 1 (recipe/reference/drama3-gb10.yaml "
    "max_running_requests: 1), so it never co-batches; the reference's default serving does.",
]
UNATTAINABLE = {
    "chunked_prefill_1_7_128": (
        "Not a reference schedule. sglang-omni/sglang_omni/models/fishaudio_drama3/engine_builder.py:172-176 rejects "
        "chunked_prefill_size > 0 ('a prompt must reach the tower in one extend'), and model_runner.py:289-301 stages "
        "each request's whole frame history per prefill. Running it would need a patched validator."),
    "decode_step_for_last_prompt_row": (
        "Not reachable without patching. A Drama 3 decode step's input frame is the frame the tower sampled "
        "(model_runner.py:195-204 and 206-235 write state_pool.frames from sampled/first frames; sglang_model.py:"
        "197-202 decodes from decode_buffers.frames). Its main token is drawn from main_token_candidate_ids "
        "(EOS + semantic band), but the fixture's prompt rows are text tokens with no semantic-band ids "
        "(backbone-reference-check.py prepare), so a 181-row prefill plus one decode step cannot feed the 182nd "
        "prompt row."),
}
LIMITS = [
    "One prompt, eager prefill, no CUDA graphs: the served breakable prefill graph pads to a token bucket and is not "
    "exercised.",
    "Companions are prefixes of the fixture prompt; their content does not enter the target rows' arithmetic except "
    "through shared GEMM/MoE calls.",
    "The engine uses max_running_requests=2 where the fixture used 1; the solo controls must reproduce the frozen bytes, "
    "otherwise the evidence is invalid.",
    "Checkpoint weights are not re-hashed here (cost); config, tokenizer and index hashes must equal the fixture's.",
    "Only Python-level sockets are refused off-host (network.blocked_attempts). C/C++ sockets (gloo/NCCL TCPStore, "
    "libcurl) are not covered: this host gives an unprivileged user unit no network namespace with a working "
    "loopback and ignores IPAddressDeny=/PrivateNetwork= for user units.",
    "Writes outside the output directory that remain: the guard's own evidence directory and the shared GPU lock "
    "(run-bounded-oracle.py), and anything a library writes to a hard-coded path that ignores the redirected "
    "HOME/TMPDIR/cache variables.",
]


def schedule_plan(rows: int) -> list[tuple[str, tuple[int, ...]]]:
    """Phases in run order; each is the prompt-row count of every co-batched request, in queue order."""
    return ([("solo-first", (rows,))]
            + [(f"companion-{k}-before", (k, rows)) for k in COMPANION_ROWS]
            + [(f"companion-{k}-after", (rows, k)) for k in COMPANION_ROWS]
            + [("pair", (rows, rows)), ("solo-last", (rows,))])


def target_views(name: str, counts: tuple[int, ...], rows: int) -> dict[str, dict]:
    """Every request carrying the whole prompt is a target view; its rows sit at its offset in the call."""
    offsets = np.cumsum((0, *counts))[:-1].tolist()
    slots = [i for i, count in enumerate(counts) if count == rows]
    return {name if len(slots) == 1 else f"{name}-{slot}": {"slot": slot, "row_offset": offsets[slot],
                                                            "call_rows": sum(counts)} for slot in slots}


def gate_results(hidden: dict, semantic: dict) -> dict:
    """The pass conjunctions of helper.hidden_metrics and helper.semantic_metrics, split per THRESHOLDS key."""
    t = helper.THRESHOLDS
    gates = {
        "hidden_min_cosine_all_rows": hidden["all_rows"]["cosine"] >= t["hidden_min_cosine_all_rows"],
        "hidden_max_relative_l2_all_rows": hidden["all_rows"]["relative_l2"] <= t["hidden_max_relative_l2_all_rows"],
        "hidden_min_row_cosine": hidden["rows"]["min_cosine"] >= t["hidden_min_row_cosine"],
        "hidden_min_last_row_cosine": hidden["last_row"]["cosine"] >= t["hidden_min_last_row_cosine"],
        "semantic_top1_equal": semantic["top1_equal"],
        "semantic_min_top30_overlap": semantic["top30_overlap"] >= t["semantic_min_top30_overlap"],
        "semantic_max_kl_reference_to_native": semantic["kl_reference_to_native"] <= t["semantic_max_kl_reference_to_native"],
        "semantic_min_cosine": semantic["cosine"] >= t["semantic_min_cosine"],
        "semantic_max_relative_l2": semantic["relative_l2"] <= t["semantic_max_relative_l2"],
    }
    if gates.keys() != t.keys() or all(gates.values()) != (hidden["pass"] and semantic["pass"]):
        raise EvidenceError("per-gate split disagrees with the helper's combined pass")
    return gates


def score(candidate: dict, reference: dict, fixture: dict) -> dict:
    """Gate metrics with ``candidate`` in the native argument slot, as helper.compare passes them."""
    case = {"hidden": helper.hidden_metrics(candidate["hidden_bits"], reference["hidden_bits"]),
            "semantic": helper.semantic_metrics(candidate["semantic"], reference["semantic"], fixture)}
    case["gates"] = gate_results(case["hidden"], case["semantic"])
    case["pass"] = all(case["gates"].values())
    return case


# ---------------------------------------------------------------- frozen inputs (CPU only)


def bind_inputs(args) -> dict:
    """Hash-check every frozen input as backbone-frozen-compare.py does; never imports torch."""
    sha = helper.sha256
    baseline_dir, native_run = args.baseline, args.native_run
    baseline = json.loads((baseline_dir / "results.json").read_text())
    fixture = json.loads((baseline_dir / "fixture.json").read_text())
    if not baseline["evidence_valid"] or not helper.THRESHOLDS == baseline["thresholds"] == fixture["thresholds"]:
        raise EvidenceError("invalid frozen baseline or changed qualification thresholds")
    if sha(HELPER_PATH) != fixture["sources_sha256"]["backbone-reference-check.py"]:
        raise EvidenceError("backbone-reference-check.py is not the code that produced the frozen fixture")
    if sha(baseline_dir / "fixture.json") != baseline["fixture_sha256"] or sha(baseline_dir / "tokens.u32le") != baseline["tokens_sha256"]:
        raise EvidenceError("frozen fixture or token hash changed")
    for name, expected in baseline["reference"]["files"].items():
        if sha(baseline_dir / name) != expected:
            raise EvidenceError(f"frozen reference artifact changed: {name}")
    checkpoint = {name: sha(args.checkpoint / file) for name, file in
                  (("config_sha256", "config.json"), ("tokenizer_sha256", "tokenizer.json"),
                   ("index_sha256", "model.safetensors.index.json"))}
    if any(fixture["checkpoint"][k] != v for k, v in checkpoint.items()):
        raise EvidenceError(f"checkpoint metadata hashes {checkpoint} differ from the fixture's")
    pin = helper.git_pin(args.reference)

    comparison = json.loads((native_run / "comparison.json").read_text())
    if not comparison["evidence_valid"] or comparison["thresholds"] != helper.THRESHOLDS \
            or comparison["frozen_results_sha256"] != sha(baseline_dir / "results.json"):
        raise EvidenceError("native comparison is invalid or was scored against other thresholds or baseline")
    for name, expected in comparison["native_files"].items():
        if sha(native_run / "native" / name) != expected:
            raise EvidenceError(f"native artifact changed: {name}")

    rows = fixture["tokens"]
    tokens = np.fromfile(baseline_dir / "tokens.u32le", dtype="<u4").astype(np.int64).tolist()
    frozen = {"hidden_bits": helper.bf16_bits(baseline_dir / "reference-hidden.bf16", (rows, helper.HIDDEN)),
              "semantic": np.fromfile(baseline_dir / "reference-semantic.f32", dtype="<f4"),
              "full_logits": np.fromfile(baseline_dir / "reference-full-logits.f32", dtype="<f4")}
    native = {"hidden_bits": helper.bf16_bits(native_run / "native/native-chunk-128-hidden.bf16", (rows, helper.HIDDEN)),
              "semantic": np.fromfile(native_run / "native/native-chunk-128-semantic.f32", dtype="<f4")}
    # Same arrays, same metric code: the recorded native comparison must come out again to the last bit.
    native_vs_frozen = score(native, frozen, fixture)
    recorded = comparison["comparisons"]["chunk-128"]
    if json.loads(json.dumps({k: native_vs_frozen[k] for k in ("hidden", "semantic")})) \
            != {k: recorded[k] for k in ("hidden", "semantic")}:
        raise EvidenceError("re-scoring the frozen native run does not reproduce its recorded comparison")
    record = {
        "script_sha256": sha(Path(__file__)), "helper": {"path": str(HELPER_PATH), "sha256": sha(HELPER_PATH)},
        "frozen_reference": {"directory": str(baseline_dir), "results_sha256": sha(baseline_dir / "results.json"),
                             "fixture_sha256": baseline["fixture_sha256"], "tokens_sha256": baseline["tokens_sha256"],
                             "files": baseline["reference"]["files"], "versions": baseline["reference"]["versions"],
                             "factory_arguments": baseline["reference"]["factory_arguments"],
                             "server_args_overrides": baseline["reference"]["server_args_overrides"],
                             "captures": baseline["reference"]["captures"]},
        "native": {"directory": str(native_run), "comparison_sha256": sha(native_run / "comparison.json"),
                   "files": comparison["native_files"],
                   "scored_files": ["native-chunk-128-hidden.bf16", "native-chunk-128-semantic.f32"],
                   "rescored_matches_recorded_comparison": True},
        "checkpoint": {"path": str(args.checkpoint), "revision_recorded": fixture["checkpoint"]["revision_recorded"],
                       **checkpoint, "weights_rehashed": False},
        "reference_checkout": {**pin, "path": str(args.reference)},
    }
    return {"record": record, "fixture": fixture, "tokens": tokens, "frozen": frozen, "native": native,
            "native_vs_frozen": native_vs_frozen}


# ---------------------------------------------------------------- launch confinement


def tree_manifest(directory: Path) -> dict:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    lines = "".join(f"{p.relative_to(directory)} {helper.sha256(p)}\n" for p in files)
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "manifest_sha256": hashlib.sha256(lines.encode()).hexdigest()}


def confine_writes(out: Path, seed: Path | None) -> dict:
    """Refuse a launch whose home, temp, cache or bytecode writes could land outside ``out``."""
    cache = out / "cache"
    misplaced = {var: os.environ.get(var) for var, name in CACHE_ENV.items() if os.environ.get(var) != str(cache / name)}
    # GIT_OPTIONAL_LOCKS=0 stops helper.git_pin's `git status` from refreshing the reference checkout's index.
    if misplaced or not sys.flags.dont_write_bytecode or os.environ.get("GIT_OPTIONAL_LOCKS") != "0":
        raise EvidenceError(f"launch is not confined to {out}: misplaced {misplaced}, dont_write_bytecode="
                            f"{sys.flags.dont_write_bytecode}, GIT_OPTIONAL_LOCKS={os.environ.get('GIT_OPTIONAL_LOCKS')!r}")
    seeded = {}
    for name in CACHE_ENV.values():
        if seed is not None and name in SEEDED_CACHES:
            shutil.copytree(seed / name, cache / name, symlinks=True)
            seeded[name] = {"source": str(seed / name), **tree_manifest(cache / name)}
        else:
            (cache / name).mkdir(parents=True)
    return {"environment": {var: os.environ[var] for var in CACHE_ENV}, "seeded_before_run": seeded}


def forbid_remote_network(blocked: list) -> None:
    """Refuse, and record, every Python-level socket operation addressed off this host."""
    local_names = {None, "", "localhost", socket.gethostname()}

    def is_local(host) -> bool:
        host = host.decode() if isinstance(host, bytes) else host
        try:
            address = ipaddress.ip_address(host.split("%")[0])
        except (AttributeError, ValueError):
            return host in local_names
        return address.is_loopback or address.is_unspecified

    def hook(event: str, args: tuple) -> None:
        if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
            # AF_UNIX paths and sendmsg on an already-connected socket carry no host.
            host = args[1][0] if isinstance(args[1], tuple) else None
        elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
            host = args[0]
        else:
            return
        if not is_local(host):
            blocked.append({"event": event, "host": repr(host)})
            raise PermissionError(f"refself refuses off-host network access: {event} {host!r}")

    sys.addaudithook(hook)


def finalize(out: Path, guard: Path) -> int:
    """A run is valid only if its results and the guard's completion record both say so."""
    final_path = out / "final.json"
    if final_path.exists():
        raise SystemExit(f"refusing to overwrite {final_path}")
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    completion = json.loads((guard / "completion.json").read_text())
    checks = {
        "results_present": bool(results),
        "results_evidence_valid": results.get("evidence_valid") is True,
        "results_from_this_script": results.get("inputs", {}).get("script_sha256") == helper.sha256(Path(__file__)),
        "guard_ran_this_script_on_this_output": str(Path(__file__).resolve()) in completion["command"]
                                                and str(out) in completion["command"],
        # Same completion contract as backbone-frozen-compare.py.
        "guard_returncode_0": completion["returncode"] == 0,
        "guard_no_failures": completion["failures"] == [],
        "guard_unit_inactive": completion["stopped_unit_state"].get("ActiveState") == "inactive",
    }
    summary = results.get("summary", {})
    final = {"schema": f"{SCHEMA}-final", "evidence_valid": all(checks.values()), "checks": checks,
             "every_alternative_view_passes_every_gate": summary.get("every_alternative_view_passes_every_gate"),
             "failed_gates_by_alternative_view": summary.get("failed_gates_by_alternative_view"),
             "results_sha256": helper.sha256(results_path) if results else None,
             "guard_completion_sha256": helper.sha256(guard / "completion.json"),
             "guard_session_sha256": helper.sha256(guard / "session.json")}
    helper.write_json(final_path, final)
    print(json.dumps({**final, "final_sha256": helper.sha256(final_path)}), flush=True)
    return 0 if final["evidence_valid"] else 1


# ---------------------------------------------------------------- reference (GPU, real scheduler)


class PhaseWatchdog(threading.Thread):
    """helper.LoopWatchdog for every request of one phase."""

    def __init__(self, scheduler, request_ids: list[str], deadline_s: float) -> None:
        super().__init__(name="refself-watchdog", daemon=True)
        self.scheduler, self.request_ids, self.deadline_s, self.reason = scheduler, set(request_ids), deadline_s, None
        self.finish = threading.Event()

    def run(self) -> None:
        end = time.monotonic() + self.deadline_s
        while not self.finish.is_set():
            outbox = self.scheduler.outbox
            with outbox.mutex:
                terminal = {m.request_id: m.type for m in outbox.queue
                            if m.request_id in self.request_ids and m.type in ("result", "error")}
            if "error" in terminal.values() or terminal.keys() == self.request_ids or time.monotonic() > end:
                self.reason = ("error" if "error" in terminal.values() else
                               "result" if terminal.keys() == self.request_ids else "timeout")
                self.scheduler.running = False
                return
            time.sleep(0.01)


def run_reference(args, out: Path, bound: dict, record: dict) -> dict:
    """helper.run_reference with two request slots and several phases."""
    import torch

    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "reference.log"), logging.StreamHandler(sys.stderr)])
    main_thread = threading.get_ident()
    total = torch.cuda.get_device_properties(0).total_memory  # first CUDA use: this thread owns the context
    torch.cuda.set_per_process_memory_fraction(min(1.0, helper.REFERENCE_ALLOCATOR_CAP_BYTES / total), 0)
    record["device"] = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                        "total_memory_bytes": total, "allocator_cap_bytes": helper.REFERENCE_ALLOCATOR_CAP_BYTES,
                        "mem_get_info_before": list(torch.cuda.mem_get_info(0))}

    import sglang_omni
    from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
    from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM

    for module in (sglang_omni, sys.modules[Drama3ForCausalLM.__module__]):
        if not Path(module.__file__).resolve().is_relative_to(args.reference.resolve()):
            raise EvidenceError(f"{module.__name__} imported from {module.__file__}, not the pinned checkout")
    record["versions"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "sglang": importlib.metadata.version("sglang"), "sglang_omni": sglang_omni.__version__,
                          "triton": importlib.metadata.version("triton")}
    record["versions_match_frozen"] = record["versions"] == bound["record"]["frozen_reference"]["versions"]
    record["environment"] = {k: v for k, v in os.environ.items()
                             if k.startswith(("SGLANG", "CUDA", "TRITON", "PYTORCH", "NCCL")) or k == "PYTHONPATH"}
    factory = {**bound["record"]["frozen_reference"]["factory_arguments"], "max_running_requests": MAX_RUNNING_REQUESTS}
    overrides = bound["record"]["frozen_reference"]["server_args_overrides"]
    record["factory_arguments"], record["server_args_overrides"] = factory, overrides
    started = time.monotonic()
    engine = [Drama3EngineBuilder(**factory).build(str(args.checkpoint), device="cuda", gpu_id=0, dtype="bfloat16",
                                                   server_args_overrides=overrides)]
    record["build_s"] = time.monotonic() - started
    try:
        return run_phases(args, out, bound, record, engine[0], main_thread)
    finally:
        helper.teardown(engine, record)


def run_phases(args, out: Path, bound: dict, record: dict, scheduler, main_thread: int) -> dict:
    import torch

    from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM

    runner = scheduler.tp_worker.model_runner
    model = runner.model
    fixture = bound["fixture"]
    if type(model) is not Drama3ForCausalLM or model.backbone_logits.__func__ is not Drama3ForCausalLM.backbone_logits:
        raise EvidenceError(f"engine model is {helper.class_name(model)}, not the pinned Drama3ForCausalLM")
    helper.write_json(out / "reference-server-args.json", helper.plain(scheduler.server_args.resolved_dict()))
    record["schedule_policy"] = getattr(scheduler.server_args, "schedule_policy", "<absent>")
    candidates = model.audio_decoder.main_token_candidate_ids.cpu().tolist()
    if candidates != [fixture["eos"], *range(fixture["semantic_start"], fixture["semantic_end"] + 1)]:
        raise EvidenceError("reference main_token_candidate_ids are not EOS followed by the semantic band")
    record["phases"], views = {}, {}
    for name, counts in schedule_plan(fixture["tokens"]):
        phase = record["phases"][name] = {"requests": list(counts)}
        captured = run_phase(args, scheduler, model, name, counts, bound, phase, main_thread)
        hidden, logits = captured["hidden"], captured["logits"]
        for view, layout in target_views(name, counts, fixture["tokens"]).items():
            start = layout["row_offset"]
            full = logits[layout["slot"]]
            views[view] = {"layout": layout, "hidden_bits": hidden[start:start + fixture["tokens"]],
                           "semantic": full[np.asarray(candidates)], "full_logits": full}
            arrays = (("hidden.bf16", views[view]["hidden_bits"], "<u2"), ("semantic.f32", views[view]["semantic"], "<f4"),
                      ("full-logits.f32", full, "<f4"))
            for suffix, array, dtype in arrays:
                array.astype(dtype).tofile(out / f"{view}-{suffix}")
            views[view]["files"] = {f"{view}-{suffix}": helper.sha256(out / f"{view}-{suffix}") for suffix, _, _ in arrays}
        phase["memory"] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                           "max_reserved_bytes": torch.cuda.max_memory_reserved(0),
                           "mem_get_info": list(torch.cuda.mem_get_info(0))}
    try:
        record.update(helper.describe_runtime(scheduler, runner, model))
    except Exception as exc:  # reported in warnings; does not alter the captured tensors
        record["runtime_description_error"] = f"{type(exc).__name__}: {exc}"
    return views


def run_phase(args, scheduler, model, name: str, counts: tuple[int, ...], bound: dict, phase: dict,
              main_thread: int) -> dict:
    """One scheduler loop over co-queued requests; the capture mirrors helper.capture_request."""
    import torch

    from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationInput, Drama3GenerationResult
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.scheduling.message import IncomingMessage

    tokens, sampling = bound["tokens"], bound["fixture"]["sampling"]
    codebooks = model.config.audio_decoder_config.num_codebooks
    captures = []
    original = model.backbone_logits

    def capturing_backbone_logits(input_ids, positions, forward_batch, frames):
        hidden_states, logits_output = original(input_ids, positions, forward_batch, frames)
        logits = logits_output.next_token_logits
        seq_lens = forward_batch.extend_seq_lens
        captures.append({
            "thread": threading.get_ident(), "mode": forward_batch.forward_mode.name,
            "rids": list(getattr(forward_batch, "rids", None) or []), "batch_size": forward_batch.batch_size,
            "extend_num_tokens": getattr(forward_batch, "extend_num_tokens", None),
            "extend_seq_lens": None if seq_lens is None else seq_lens.detach().clone(),
            "input_ids": input_ids.detach().clone(), "positions": positions.detach().clone(),
            "frames": frames.detach().clone(), "hidden": hidden_states.detach().clone(),
            "logits": None if logits is None else logits.detach().clone(),
        })
        return hidden_states, logits_output

    request_ids = [f"refself-{name}-{slot}" for slot in range(len(counts))]
    messages = []
    for request_id, count in zip(request_ids, counts):
        prompt_rows = torch.zeros((count, 1 + codebooks), dtype=torch.int64)
        prompt_rows[:, 0] = torch.tensor(tokens[:count], dtype=torch.int64)
        payload = StagePayload(request_id=request_id, request=OmniRequest(inputs=None, params={}), data=Drama3GenerationInput(
            prompt_rows=prompt_rows, max_new_tokens=helper.MAX_NEW_TOKENS, temperature=sampling["temperature"],
            top_k=sampling["top_k"], top_p=sampling["top_p"]))
        messages.append(IncomingMessage(request_id=request_id, type="new_request", data=payload))
    model.backbone_logits = capturing_backbone_logits
    # Queued before the loop starts, so its first inbox drain admits every request of the phase together.
    for message in messages:
        scheduler.inbox.put(message)
    watchdog = PhaseWatchdog(scheduler, request_ids, args.reference_loop_timeout)
    started = time.monotonic()
    watchdog.start()
    try:
        scheduler.start()  # the real OmniScheduler loop, on this (CUDA-owning) thread
    finally:
        watchdog.finish.set()
        watchdog.join()
        del model.backbone_logits  # restore the class method
    phase["loop_s"] = time.monotonic() - started
    phase["loop_stop_reason"] = watchdog.reason
    outbox = []
    while not scheduler.outbox.empty():
        outbox.append(scheduler.outbox.get_nowait())
    phase["outbox"] = [{"request_id": m.request_id, "type": m.type,
                        "data": repr(m.data) if m.type == "error" else helper.class_name(m.data)} for m in outbox]
    results = {m.request_id: m.data.data for m in outbox if m.type == "result"}
    if watchdog.reason != "result" or results.keys() != set(request_ids) or any(m.type == "error" for m in outbox):
        raise EvidenceError(f"{name}: scheduler loop ended with {watchdog.reason}; outbox {phase['outbox']}")
    if any(not isinstance(results[r], Drama3GenerationResult) or results[r].prompt_row_count != c
           for r, c in zip(request_ids, counts)):
        raise EvidenceError(f"{name}: unexpected terminal payloads")

    phase["captures"] = [{k: v for k, v in c.items() if not isinstance(v, torch.Tensor)} for c in captures]
    if [c["mode"] for c in captures] != ["EXTEND"]:
        raise EvidenceError(f"{name}: expected one co-batched EXTEND; got modes {[c['mode'] for c in captures]}")
    capture = captures[0]
    call_rows = sum(counts)
    expected_ids = torch.tensor([t for count in counts for t in tokens[:count]], dtype=torch.int64)
    hidden, logits = capture["hidden"], capture["logits"]
    frames = capture["frames"].cpu().long()
    checks = {
        "same_thread": capture["thread"] == main_thread,
        "request_ids_in_queue_order": capture["rids"] == request_ids and capture["batch_size"] == len(counts),
        "extend_rows": capture["extend_num_tokens"] == call_rows
                       and capture["extend_seq_lens"] is not None and capture["extend_seq_lens"].tolist() == list(counts),
        "input_ids": torch.equal(capture["input_ids"].cpu().long(), expected_ids),
        "positions": torch.equal(capture["positions"].cpu().long(), torch.cat([torch.arange(c) for c in counts])),
        "text_frames": tuple(frames.shape) == (call_rows, 1 + codebooks) and torch.equal(frames[:, 0], expected_ids)
                       and not bool(frames[:, 1:].any()),
        "hidden_shape": tuple(hidden.shape) == (call_rows, helper.HIDDEN) and hidden.dtype == torch.bfloat16,
        "logits_shape": logits is not None and tuple(logits.shape) == (len(counts), model.logits_processor.vocab_size)
                        and logits.dtype == torch.float32,
    }
    phase["evidence_checks"] = checks
    if not all(checks.values()):
        raise EvidenceError(f"{name}: capture checks failed: {[k for k, v in checks.items() if not v]}")
    # SGLang prunes the extend to each request's last row (logits_processor.py:677-678) before the lm_head;
    # rerun its own _compute_lm_head on the same rows, as helper.capture_request does for one.
    if model.logits_processor.logit_scale is not None or model.logits_processor.final_logit_softcapping:
        raise EvidenceError("logit scaling or softcapping present; the frozen run had neither")
    last_rows = torch.cumsum(capture["extend_seq_lens"].long(), 0) - 1
    with torch.inference_mode():
        again = model.logits_processor._compute_lm_head(hidden[last_rows], model.lm_head)
    phase["lm_head_input_consistency"] = bool(torch.equal(again[:, :logits.shape[1]].float(), logits))
    if not phase["lm_head_input_consistency"]:
        raise EvidenceError(f"{name}: captured hidden is not the lm_head input of the captured logits")
    torch.cuda.synchronize()
    hidden_bits = hidden.view(torch.int16).cpu().numpy().view(np.uint16).reshape(call_rows, helper.HIDDEN)
    full_logits = logits.cpu().numpy()
    if not np.isfinite(helper.bits_to_f32(hidden_bits)).all() or not np.isfinite(full_logits).all():
        raise EvidenceError(f"{name}: hidden or logits are not finite")
    captures.clear()
    return {"hidden": hidden_bits, "logits": full_logits}


# ---------------------------------------------------------------- comparison


def compare(views: dict, bound: dict) -> tuple[dict, dict]:
    fixture, frozen, native = bound["fixture"], bound["frozen"], bound["native"]
    comparisons = {}
    for view, data in views.items():
        comparisons[view] = {
            "layout": data["layout"], "files": data["files"],
            "bit_exact_to_frozen": {
                "hidden": bool(np.array_equal(data["hidden_bits"], frozen["hidden_bits"])),
                "semantic": data["semantic"].astype("<f4").tobytes() == frozen["semantic"].tobytes(),
                "full_logits": data["full_logits"].astype("<f4").tobytes() == frozen["full_logits"].tobytes()},
            "schedule_vs_frozen_reference": score(data, frozen, fixture),
            "native_vs_schedule": score(native, data, fixture),
        }
    controls = {v: all(comparisons[v]["bit_exact_to_frozen"].values()) for v in CONTROL_PHASES}
    alternatives = [v for v in comparisons if v not in CONTROL_PHASES]
    failing = {v: [g for g, ok in comparisons[v]["schedule_vs_frozen_reference"]["gates"].items() if not ok]
               for v in alternatives}
    summary = {
        "controls_bit_exact": controls,
        "alternative_views": alternatives,
        "alternative_views_bit_exact_to_frozen": [v for v in alternatives
                                                  if all(comparisons[v]["bit_exact_to_frozen"].values())],
        "failed_gates_by_alternative_view": {v: g for v, g in failing.items() if g},
        "every_alternative_view_passes_every_gate": not any(failing.values()),
        "native_passes_against_views": [v for v in comparisons if comparisons[v]["native_vs_schedule"]["pass"]],
    }
    for view in comparisons:
        case = comparisons[view]["schedule_vs_frozen_reference"]
        print(json.dumps({"view": view, **comparisons[view]["layout"], "pass": case["pass"],
                          "hidden_relative_l2": case["hidden"]["all_rows"]["relative_l2"],
                          "hidden_min_row_cosine": case["hidden"]["rows"]["min_cosine"],
                          "semantic_kl": case["semantic"]["kl_reference_to_native"]}), flush=True)
    return comparisons, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, nargs="?", help="new evidence directory (must not exist)")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, help="sglang-omni checkout at the fixture's pinned revision")
    parser.add_argument("--baseline", type=Path, help="frozen backbone-reference-check.py output")
    parser.add_argument("--native-run", type=Path, help="backbone-frozen-compare.py native run, scored for context")
    parser.add_argument("--seed-cache", type=Path, help="runtime-cache of an earlier guarded run; triton/ and cuda/ are copied")
    parser.add_argument("--reference-loop-timeout", type=float, default=120.0, help="seconds per scheduler phase")
    parser.add_argument("--plan-only", action="store_true", help="check frozen inputs on the CPU, print the plan, exit")
    parser.add_argument("--finalize", type=Path, metavar="GUARD",
                        help="after the guard exits: bind OUTPUT/results.json to GUARD/completion.json in final.json")
    args = parser.parse_args()
    if args.output is None and not args.plan_only:
        parser.error("the output directory is required unless --plan-only is given")
    if args.finalize is not None:
        return finalize(args.output.resolve(), args.finalize.resolve())
    if None in (args.checkpoint, args.reference, args.baseline, args.native_run):
        parser.error("--checkpoint, --reference, --baseline and --native-run are required")
    if args.plan_only:
        bound = bind_inputs(args)
        rows = bound["fixture"]["tokens"]
        print(json.dumps(helper.plain({
            "inputs": bound["record"], "max_running_requests": MAX_RUNNING_REQUESTS,
            "phases": [{"phase": name, "requests": list(counts), "views": target_views(name, counts, rows)}
                       for name, counts in schedule_plan(rows)],
            "native_vs_frozen_pass": bound["native_vs_frozen"]["pass"], "unattainable": UNATTAINABLE,
            "torch_imported": "torch" in sys.modules}), indent=2))
        return 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    blocked = []
    forbid_remote_network(blocked)
    result = {"schema": SCHEMA, "scope": SCOPE, "thresholds": helper.THRESHOLDS, "evidence_valid": False,
              "errors": [], "warnings": [], "limits": LIMITS, "serving_evidence": SERVING_EVIDENCE,
              "unattainable": UNATTAINABLE, "max_running_requests": MAX_RUNNING_REQUESTS,
              "network": {"python_off_host_refused": True, "blocked_attempts": blocked}, "reference": {}}
    try:
        result["confinement"] = confine_writes(out, args.seed_cache)
        bound = bind_inputs(args)
        result["inputs"] = bound["record"]
        result["native_vs_frozen"] = bound["native_vs_frozen"]
        views = run_reference(args, out, bound, result["reference"])
        result["comparisons"], result["summary"] = compare(views, bound)
        failed_controls = [v for v, ok in result["summary"]["controls_bit_exact"].items() if not ok]
        if failed_controls:
            raise EvidenceError(f"solo controls {failed_controls} do not reproduce the frozen reference bytes")
        result["evidence_valid"] = True
    except EvidenceError as exc:
        result["errors"].append(f"EvidenceError: {exc}")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        logging.getLogger(__name__).exception("reference self-control failed")
    reference_record = result["reference"]
    if "runtime_description_error" in reference_record:
        result["warnings"].append(f"runtime settings not recorded: {reference_record['runtime_description_error']}")
    if reference_record.get("versions_match_frozen") is False:
        result["warnings"].append("library versions differ from the frozen reference run")
    result["warnings"] += [f"teardown {e}" for e in reference_record.get("teardown_errors", [])]
    helper.write_json(out / "results.json", helper.plain(result))
    print(json.dumps({"evidence_valid": result["evidence_valid"], "errors": result["errors"],
                      "every_alternative_view_passes_every_gate":
                          result.get("summary", {}).get("every_alternative_view_passes_every_gate"),
                      "results_sha256": helper.sha256(out / "results.json")}), flush=True)
    return 0 if result["evidence_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
