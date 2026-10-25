#!/usr/bin/env bash
# Before/after speed benchmark: the sglang-omni reference, then TensorFold native, one engine at a time under
# recipe/pilot.py. Each engine gets whole WAV replies on a fresh process (1 cold, 2 warmup, 3 warm passes of the
# 24-item corpus), then streamed PCM on the same process (2 warmup, 3 warm passes). Ends with bench/tables.py.
#
# Environment (all required unless a default is shown):
#   CHECKPOINT    fishaudio/drama-3 checkpoint directory
#   RUNTIME_DIR   pilot runtime directory; holds venv/ and sglang-omni/ for the reference
#   REFERENCE_SRC sglang-omni checkout the reference runs from (default: RUNTIME_DIR/sglang-omni)
#   ENGINE_DIR    patched TensorFold tree built by scripts/build-engine.sh
#   GPU_LOCK      the host's existing GPU lock file
#   OUT           new output directory
#   LIBRARY_DIR   optional CUDA library directory for the servers
#   PORT          loopback port (default 18793)
set -uo pipefail

: "${CHECKPOINT:?}" "${RUNTIME_DIR:?}" "${ENGINE_DIR:?}" "${GPU_LOCK:?}" "${OUT:?}"
REFERENCE_SRC=${REFERENCE_SRC:-$RUNTIME_DIR/sglang-omni}
PORT=${PORT:-18793}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
recipe=$repo/recipe
native=$ENGINE_DIR/zig-out/native/bin/tensorfold-native
library=()
[[ -n ${LIBRARY_DIR:-} ]] && library=(--library-dir "$LIBRARY_DIR")

[[ -x $native ]] || { echo "missing $native; run scripts/build-engine.sh" >&2; exit 1; }
mkdir "$OUT" || exit 1
mkdir "$OUT/runs" "$OUT/identity"

ref_precision='reference/drama3-gb10.yaml: Triton attention, FP16 codec decode (decode_dtype float16); checkpoint dtype bfloat16 with bsgemm-moe 128x128 dynamic FP8 MoE (config.json)'
native_sha=$(sha256sum "$native" | cut -c1-8)
native_precision="native CUDA kernels embedded in tensorfold-native $native_sha; checkpoint dtype bfloat16 with bsgemm-moe 128x128 dynamic FP8 MoE (config.json)"

serve() {  # serve NAME PILOT-ARGS...: starts pilot.py, waits for ready.json, sets PILOT and SERVER_PID
  local name=$1
  shift
  python3 -B "$recipe/pilot.py" "$@" --runtime-dir "$RUNTIME_DIR" --checkpoint "$CHECKPOINT" --gpu-lock "$GPU_LOCK" \
    --evidence-dir "$OUT/pilot-$name" --port "$PORT" "${library[@]}" > "$OUT/pilot-$name.stdout" 2>&1 &
  PILOT=$!
  until [[ -f $OUT/pilot-$name/ready.json ]]; do
    kill -0 "$PILOT" 2>/dev/null || { echo "pilot $name exited before ready" >&2; return 1; }
    sleep 2
  done
  local unit
  unit=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['unit'])" "$OUT/pilot-$name/ready.json")
  SERVER_PID=$(systemctl --user show --property=MainPID --value "$unit")
  echo "$name ready: unit $unit server pid $SERVER_PID"
}

bench() {  # bench RUN LABEL PRECISION SOURCE PILOT-EVIDENCE START [extra...]; exit 1 (some requests failed) is a result
  local run=$1 label=$2 precision=$3 source=$4 evidence=$5 start=$6 status
  shift 6
  python3 -B "$repo/bench/drama_bench.py" run --endpoint "http://127.0.0.1:$PORT" --model fishaudio/drama-3 \
    --corpus "$repo/bench/corpus/drama-bench-en-v1.jsonl" --evidence-dir "$OUT/runs/$run" --label "$label" \
    --precision "$precision" --server-pid "$SERVER_PID" --server-source "$source" --checkpoint "$CHECKPOINT" \
    --checkpoint-pins "$recipe/checksums/checkpoint.sha256" --pilot-evidence "$evidence" \
    --server-start "$start" --warmup 2 --repeats 3 --timeout 120 "$@" > "$OUT/runs/$run.log" 2>&1
  status=$?
  echo "$run exit=$status"
  ((status < 2))
}

stop() {
  kill -INT "$PILOT"
  wait "$PILOT"
  echo "pilot exit=$?"
}

admit() {  # admit NAME PEAK-BYTES PEAK-SOURCE
  python3 -B "$repo/bench/admit.py" --root "$CHECKPOINT" --peak-bytes "$2" --margin-bytes 4294967296 \
    --peak-source "$3" --out "$OUT/identity/admission-$1.json"
}

echo "== reference $(date -Is)"
admit reference 57982058496 'reference/drama3-gb10.yaml stage reserves: preprocessing 4 GiB + tts_engine 44 GiB + vocoder 6 GiB' || exit 1
serve reference --backend sglang-omni || exit 1
bench reference-whole "BEFORE: sglang-omni reference via pilot.py, whole WAV replies" \
  "$ref_precision" "$REFERENCE_SRC" "$OUT/pilot-reference" fresh \
  --config "$recipe/reference/drama3-gb10.yaml" || { stop; exit 1; }
bench reference-stream "BEFORE: sglang-omni reference via pilot.py, stream=true raw PCM (same process)" \
  "$ref_precision" "$REFERENCE_SRC" "$OUT/pilot-reference" reused \
  --config "$recipe/reference/drama3-gb10.yaml" --response-format pcm --stream || { stop; exit 1; }
stop

echo "== native $(date -Is)"
admit native 40857363572 'native tracked CUDA peak at /health after a serving gate on GB10' || exit 1
serve native --native-bin "$native" || exit 1
bench native-whole "AFTER: TensorFold native $native_sha via pilot.py, whole WAV replies" \
  "$native_precision" "$ENGINE_DIR" "$OUT/pilot-native" fresh || { stop; exit 1; }
bench native-stream "AFTER: TensorFold native $native_sha via pilot.py, stream=true raw PCM (same process)" \
  "$native_precision" "$ENGINE_DIR" "$OUT/pilot-native" reused --response-format pcm --stream || { stop; exit 1; }
stop

echo "== tables $(date -Is)"
python3 -B "$repo/bench/tables.py" --reference-whole "$OUT/runs/reference-whole" --native-whole "$OUT/runs/native-whole" \
  --reference-stream "$OUT/runs/reference-stream" --native-stream "$OUT/runs/native-stream" --reference-streams \
  --out "$OUT/tables.json"
