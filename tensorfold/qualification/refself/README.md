# refself: reference-vs-reference control for the backbone gates

`refself.py` asks whether the pinned SGLang-Omni reference passes the native full-backbone gates against its own
frozen output (the `--baseline` fixture from `backbone-reference-check.py`) when only the prefill call schedule
changes. The native comparison's thresholds stay unchanged.

## What runs

It uses the same path that produced the fixture: `Drama3EngineBuilder` → the real `OmniScheduler` loop, eager,
with `Drama3ForCausalLM.backbone_logits` wrapped on the instance (`backbone-reference-check.py` `run_reference`
and `capture_request`). The factory arguments and overrides are the frozen run's, except
`max_running_requests=2`. One engine runs these phases in order, each a single EXTEND:

| Phase | Requests (rows, queue order) | Rows in the call | Target offset |
|---|---|---|---|
| solo-first / solo-last | 182 | 182 | 0 (determinism controls; must equal the frozen bytes) |
| companion-{2,7,128}-before | k, 182 | 184 / 189 / 310 | 2 / 7 / 128 |
| companion-{2,7,128}-after | 182, k | 184 / 189 / 310 | 0 |
| pair | 182, 182 | 364 | 0 and 182 (views `pair-0`, `pair-1`) |

Companions are prefixes of the fixture prompt. The reference refuses prompts under two rows
(`request_builders.py:24-25`), so the smallest companion is 2 rows.

The reference's served defaults co-batch: `DEFAULT_MAX_RUNNING_REQUESTS = 16` (`config.py:19`). The recipe runs
the reference at concurrency 1 and never co-batches. Two schedules cannot be run without patching the reference,
so they are reported as unattainable (`results.json` → `unattainable`):

- **Chunked prefill 1/7/128.** `engine_builder.py:172-176` rejects it.
- **An M=1 decode step for the last prompt row.** Decode inputs are frames the tower sampled.

Every target view is scored against the frozen reference with `hidden_metrics`/`semantic_metrics`, imported
unchanged from `backbone-reference-check.py`, and the thresholds are its `THRESHOLDS`, which must equal the
frozen baseline's. The native run (`--native-run`, a `backbone-frozen-compare.py` output) is scored against every
view for context.

## Commands

`R` is a new run root. The guard creates `R/guard`, and the script creates `R/out`. Paths below are absolute,
because `--finalize` checks that the guard ran this script on this output.

CPU-only check of every frozen input (no CUDA, no bytecode, no index refresh):

```sh
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES= "$RUNTIME_DIR/venv/bin/python" -B \
  "$REPO/qualification/refself/refself.py" --plan-only \
  --checkpoint "$CHECKPOINT" --reference "$RUNTIME_DIR/sglang-omni" --baseline "$BASELINE" --native-run "$NATIVE_RUN"
```

1. The guarded GPU run. The outer `PYTHONDONTWRITEBYTECODE=1 ... -B` stops the guard interpreter, which imports
   `recipe/pilot.py`, from writing bytecode. The inner `/usr/bin/env` points every cache into `R/out/cache`,
   overriding the guard's runtime-directory caches:

   ```sh
   C="$R/out/cache"
   PYTHONDONTWRITEBYTECODE=1 python3 -B "$REPO/qualification/run-bounded-oracle.py" \
     --gpu-lock "$GPU_LOCK" --runtime-dir "$RUNTIME_DIR" "$R/guard" \
     /usr/bin/env HOME="$C/home" XDG_CACHE_HOME="$C/xdg" TMPDIR="$C/tmp" TRITON_CACHE_DIR="$C/triton" \
       CUDA_CACHE_PATH="$C/cuda" TORCHINDUCTOR_CACHE_DIR="$C/inductor" FLASHINFER_WORKSPACE_BASE="$C/flashinfer" \
       HF_HOME="$C/hf" GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 \
     "$RUNTIME_DIR/venv/bin/python" -B "$REPO/qualification/refself/refself.py" "$R/out" \
       --checkpoint "$CHECKPOINT" --reference "$RUNTIME_DIR/sglang-omni" --baseline "$BASELINE" \
       --native-run "$NATIVE_RUN" --seed-cache "$RUNTIME_DIR/runtime-cache"
   ```

2. After the guard exits, bind the guard record (CPU only):

   ```sh
   PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$RUNTIME_DIR/venv/bin/python" -B \
     "$REPO/qualification/refself/refself.py" "$R/out" --finalize "$R/guard"
   ```

**The run is valid only if `R/out/final.json` has `evidence_valid: true`** (exit 0 of `--finalize`).
`--finalize` requires:

- `results.json` from this script's hash, with `evidence_valid`: every phase was captured and checked, and both
  solo controls are bit-exact to the frozen bytes;
- the guard's `completion.json` shows this script on this output, return code 0, no failures and an inactive
  unit.

A `results.json` on its own proves nothing: the guard can kill the process after the file is written. Valid
evidence does not mean the gates passed; `final.json` reports the outcome in
`every_alternative_view_passes_every_gate` and `failed_gates_by_alternative_view`.

## Confinement and network (what is enforced, what is not)

**Writes.** Before it touches inputs, the full mode refuses to start unless every one of `HOME`,
`XDG_CACHE_HOME`, `TMPDIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `TORCHINDUCTOR_CACHE_DIR`,
`FLASHINFER_WORKSPACE_BASE` and `HF_HOME` points to `OUTPUT/cache/<name>`. It also requires bytecode writes off
(`-B`) and `GIT_OPTIONAL_LOCKS=0`, so the helper's `git status` does not refresh the reference checkout's index.

With `--seed-cache`, its `triton/` and `cuda/` are *copied* into `OUTPUT/cache` and recorded as a
file-count/byte/sha256 manifest in `results.json.confinement`, so kernels an earlier guarded run compiled are
reused. The solo controls must still reproduce the frozen bytes.

Writes outside `OUTPUT` that remain: the guard's `R/guard` evidence, its `flock` on the shared GPU lock, the
systemd journal, and any library path hard-coded to ignore the redirected variables.

**Network.** Unprivileged user units usually cannot get a network namespace with a working loopback, and
`systemd-run --user` ignores `IPAddressDeny=` and `PrivateNetwork=`. What is enforced: a Python audit hook
(`forbid_remote_network`) refuses, and records in `results.json.network.blocked_attempts`, every Python-level
`connect`/`sendto`/`sendmsg` to a non-loopback address, and name lookups of anything but localhost or this host's
name. What is not enforced: sockets opened from C/C++ (gloo/NCCL TCPStore, libcurl). The guard also sets
`HF_HUB_OFFLINE=1`.

## Outputs (`R/out/`)

- `results.json`: script, helper, frozen-fixture, native and checkpoint-metadata hashes; the reference git pin
  and library versions; confinement and seeded-cache manifests; blocked network attempts; per-phase capture
  checks, timings and memory; per-view `bit_exact_to_frozen`, `schedule_vs_frozen_reference` and
  `native_vs_schedule`; `native_vs_frozen`.
- `final.json`, from `--finalize`.
- `{view}-hidden.bf16`, `{view}-semantic.f32`, `{view}-full-logits.f32`.
- `reference.log`, `reference-server-args.json`.
- `cache/`.
