# Qualification tools

> **Legacy:** these gates qualify the old patched engine (commit `0f54000`, 59 patches on TensorFold `c7137bb3`),
> whose oracle build steps the upstream base does not have. Check out `0f54000` to rerun them.

The numerical gates that compare the native engine with the pinned SGLang-Omni reference
(`ab095ef9`, checkpoint `e0a9218a`). Their current results are in [`../results/`](../results/); this directory
lets you rerun them.

| Tool | Runs on | What it checks |
|---|---|---|
| `run-bounded-oracle.py` | host | The GPU guard for every GPU step below: `recipe/pilot.py`'s admission, 54 GiB cap, 34 GiB abort floor and shared lock, plus a 600 s deadline |
| `backbone-reference-check.py` | GPU | The full 40-layer backbone prefill of one 182-token prompt: native chunks 1/7/128 against one reference extend. Writes the frozen fixture the other backbone tools use |
| `backbone-frozen-compare.py` | CPU | A new native oracle run against that frozen fixture, without re-running the reference |
| `refself/refself.py` | GPU | The reference against its own frozen output under co-batched schedules ([README](refself/README.md)) |
| `worst-row-bar.py` | CPU | Derives the revised worst-row cosine bar from the refself views alone |
| `rescore-native.py` | CPU | Re-scores native backbone outputs under that bar; every other threshold unchanged |
| `tower-reference-check.py` | GPU | The 13-layer tower, teacher-forced, against the reference's unfused `AudioDecoder` at 1, 7 and 128 rows |

Thresholds are declared in the scripts before any run (`THRESHOLDS`); never relax them to make a run pass.

## Requirements

- The engine and its oracle tools, built with `scripts/build-engine.sh` (`--engine build/tensorfold`).
- The checkpoint, verified with `recipe/checksums/checkpoint.sha256`.
- A runtime directory as for the recipe's reference backend: `venv/` (Python 3.12 with the reference's
  dependencies, `numpy` and `torch`) and `sglang-omni/`, a clean checkout at `ab095ef9`.
- Run every script with that venv's Python. The CPU-only scripts need only `numpy` (and `torch` for
  `backbone-frozen-compare.py`).

## The backbone sequence

```sh
Q=qualification
G="python3 -B $Q/run-bounded-oracle.py --gpu-lock $GPU_LOCK --runtime-dir $RUNTIME_DIR"
PY=$RUNTIME_DIR/venv/bin/python

# 1. Frozen fixture: the native oracle, then the reference, in one guarded run.
$G "$RUNS/fixture-guard" $PY -B "$PWD/$Q/backbone-reference-check.py" "$RUNS/fixture" \
  --checkpoint "$CHECKPOINT" --reference "$RUNTIME_DIR/sglang-omni" --engine "$PWD/build/tensorfold"

# 2. Any later native build, compared against the frozen fixture without re-running the reference.
mkdir -p "$RUNS/native-01/native"
$G "$RUNS/native-01-guard" "$PWD/build/tensorfold/zig-out/oracle/bin/tf-drama-backbone-oracle" \
  "$CHECKPOINT" "$RUNS/fixture/tokens.u32le" "$RUNS/native-01/native"
$PY -B $Q/backbone-frozen-compare.py "$RUNS/native-01" "$RUNS/native-01-guard" \
  --baseline "$RUNS/fixture" --engine build/tensorfold

# 3. The reference against itself (see refself/README.md for the confined launch), then the bar.
$PY -B $Q/worst-row-bar.py "$RUNS/refself/out" "$RUNS/fixture"

# 4. Native under the revised bar.
$PY -B $Q/rescore-native.py "$RUNS/native-01/native" "$RUNS/fixture" "$RUNS/rescore-01.json"
```

`backbone-reference-check.py --prepare-only` writes the fixture without CUDA. The tower check is one guarded run:

```sh
$G "$RUNS/tower-guard" $PY -B "$PWD/$Q/tower-reference-check.py" "$RUNS/tower" \
  --checkpoint "$CHECKPOINT" --reference "$RUNTIME_DIR/sglang-omni" --engine "$PWD/build/tensorfold"
```

## How the worst-row bar was set

The original `hidden_min_row_cosine` bar was 0.995. The reference fails it against its own solo output in every
co-batched schedule it supports, so the bar was revised from reference data alone, without consulting native
data, and registered before native was re-scored:

- **Rule.** The bar is the lowest minimum row cosine against `solo-first` over the eight co-batched views
  (`companion-{2,7,128}-{before,after}`, `pair-0`, `pair-1`), in float64, with no margin or rounding. A candidate
  passes iff candidate ≥ bar.
- **Excluded views.** The solo views are determinism controls, not co-batched, so they do not set the bar; both
  must be byte-identical to the frozen reference. Chunked prefill and a synthetic last-row decode are excluded
  because the unmodified reference cannot run them.
- **Value.** 0.8712518909203973, set by `companion-128-before`, `companion-128-after` and `pair-0`, which share
  identical hidden bytes.
- **Limits.** One 182-token fixture, eager execution without CUDA graphs, eight two-request views and a single
  run with no repeat-variance estimate. A bar this low permits a substantial isolated row discrepancy, so every
  other gate stays unchanged.

`worst-row-bar.py` reproduces the derivation from a refself run and reports whether it equals the registered
value. A refself run on other hardware or library builds may give a different envelope.

## Not included

These gate tools are not in this curated set:

- the codec gate (`1, 2, 7, 128, 1024` frames against the reference's `ChannelsLastDecoder`, 40 dB SNR),
- its chunked-versus-whole streaming verifier,
- the tower full-context and real-input replays.

Their inputs are frozen artifacts from earlier runs, pinned by hash at design time, and making them reproducible
from this repository would mean rewriting them.
