# macOS: running the pinned recipe on Apple silicon

The full-model Metal build uses the same pinned TensorFold commit, ordered patch series,
checkpoint revision and HTTP contract as the Linux recipe. It does not require CUDA,
`nvcc`, systemd, or `/proc`. Use `recipe/pilot_macos.py`; the Linux launcher remains unchanged.
The separate NVFP4 engine/weights are not qualified by this Mac guide.

This is a local, cooperative pilot—not a production service installer, GPU reservation
system, memory sandbox, or proof that the whole host is idle. Do not interrupt another
application to make an admission check pass.

## Prerequisites and build

- Apple silicon with Xcode command-line tools, including `xcrun metal`.
- Zig **0.17.0**. The official `zig-aarch64-macos-0.17.0.tar.xz` archive has SHA-256
  `b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a`.
- Python 3.9 or newer for the stdlib-only launcher and validation tools.
- Approximately 39 GiB of checkpoint storage plus build/cache space, and sufficient
  free + inactive memory for admission. Disk size is not a resident-memory prediction.
- Permission to use the checkpoint under the [pinned model license](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE).

From the repository root:

```sh
hf download fishaudio/drama-3 --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 --local-dir drama-3
(cd drama-3 && shasum -a 256 -c ../recipe/checksums/checkpoint.sha256)
ZIG=/absolute/path/to/zig-0.17.0/zig scripts/build-engine.sh --weights full
```

All ten checkpoint files must report `OK` before a model run. The build script checks the
ordered patches and omits the Linux/Windows-only `-Dnvcc` option on Darwin. Outputs:

- `build/tensorfold/zig-out/native/bin/tensorfold-native`
- `build/tensorfold/zig-out/bin/drama3-generate`

Use the installed build's `serve --help` for its supported options; do not assume features
from newer upstream releases exist in this pinned tree.

## Launch and stop

First coordinate with other GPU users. Use the **same agreed lock file** as other cooperating
launchers; a newly invented private lock does not exclude unrelated workloads. For an otherwise
unshared local pilot, the following creates its local runtime directory and lock:

```sh
mkdir -p runtime
touch runtime/gpu.lock
python3 recipe/pilot_macos.py \
  --native-bin build/tensorfold/zig-out/native/bin/tensorfold-native \
  --checkpoint drama-3 --runtime-dir runtime --gpu-lock runtime/gpu.lock \
  --evidence-dir runs/macos-pilot-01
```

The default bind address is **127.0.0.1**, port **18793**; `--host` accepts numeric IPv4 addresses. An occupied port or lock is a refusal,
not permission to displace its owner. This unauthenticated API should not be exposed casually;
LAN placement does not exempt a hosted service from the model license.

The launcher records `session.json`, `server.log`, `memory.jsonl`, `ready.json` and
`completion.json`. `DRAMA_PILOT_READY` means the launched server passed its health check, not
that audio correctness or performance has been verified. Explicit `--evidence-reuse` clears stale
readiness after acquiring the lock, even when admission then fails. Without it an existing evidence
directory is refused. Prefer a new evidence directory for each attempt.

| Guard | macOS behavior |
|---|---|
| Admission | max(checkpoint bytes + 30 GiB, 34 GiB) of free + inactive memory |
| Memory observation | One `vm_stat` snapshot per sample; free + inactive only |
| Abort floor | 34 GiB; admission overrides cannot be lower |
| Isolation | Owned child process group and inherited cooperative lock |
| Memory cap | None: Metal has no equivalent of the Linux cgroup/CUDA cap |

`vm_stat` speculative pages are already included in free pages. Purgeable pages are not added;
these categories must not be counted twice. `--min-available-gib` is an explicit admission
override, not a memory cap; it must be finite and at least the abort floor. Other numeric
controls must also be finite and valid.

Stop the launcher with Ctrl-C or SIGTERM. It settles only its own process group, with bounded
TERM/KILL cleanup, and retains the unreaped leader identity while checking descendants.
An unexpected server exit is a failure even when the server's exit status is zero. Observation
errors fail closed. The server inherits the lock descriptor, so killing only the launcher
cannot release the lock while that server is still running. This is not an OS sandbox:
escaped descendants or unkillable processes require operator attention and must not be reported
as settled. Read `completion.json` before starting another attempt.

No LaunchAgent is installed by this repository. Do not copy unrelated host-specific service
names or restart commands into an automated setup.

## Speech requests

```sh
curl --fail-with-body -sS -D headers.txt http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fishaudio/drama-3","input":"The train leaves at seven fifteen."}' \
  -o hello.wav
afplay hello.wav
```

For streaming, add `"response_format":"pcm","stream":true` and use `curl -N`. The body is
mono signed 16-bit little-endian PCM at 44.1 kHz; `afplay` needs a suitable container such as WAV.
Check the HTTP status and `X-Finish-Reason` rather than treating any saved file as successful audio.

## Attested reference voices

Use a synthetic voice, your own voice, or a voice for which you have the required permission.
The caller must attest the source explicitly; the catalog writer does not invent consent.
Reference audio, codec rows, and generated voice catalogs stay out of Git.

Encode a matching transcript and a clean single-speaker reference clip:

```sh
build/tensorfold/zig-out/bin/drama3-generate --codec drama-3/codec.pth \
  --encode clip.codes --ref-audio clip.wav
python3 recipe/build_voice.py --source self my-voice \
  "My own recording; permission retained locally." "Exact transcript of clip.wav:clip.codes"
```

`--source` is required and accepts `synthetic`, `self`, or `consented`. `--output-dir` defaults
to `runtime/voices`. The tool refuses to overwrite an existing catalog and validates the server's
contract before publishing a file:

- ASCII name of 1–64 bytes, beginning with a letter/digit, then letters/digits/`-`/`_`;
  `default` is reserved, case-insensitively.
- Nonblank UTF-8 source note and transcripts, each at most 4096 bytes. Transcripts cannot
  contain structural prompt tokens or unsupported/malformed speaker tags.
- One to four references; each has 1–646 complete rows of ten little-endian int32 codes.
- Codebook zero: 0–4095; codebooks one through nine: 0–1023; catalog at most 256 KiB.

Start a new coordinated pilot with `-- --voices /absolute/path/to/runtime/voices`, then request
`"voice":"my-voice"`. Paths after `--` are server arguments and are resolved from its runtime
directory. `GET /v1/voices` lists loaded voices. Without `--default-voice`, omitted/default
requests use the builtin voice. Setting `--default-voice NAME` changes both cases.

Patch 0012 reuses one reference-prefix snapshot per Metal lane. A different or absent prefix
invalidates the live cache before prefill overwrites it; switching voices reuses the allocated
snapshot buffers rather than retaining a new allocation per switch. Prefix identity includes
the prompt boundary and the audio tower's lookahead dependency. Generation settings and
sampling are unchanged. This is an output-preserving optimization, not a multi-voice cache.

## Correctness gates

CPU tests and bounded subprocess smoke are separate from model work:

```sh
python3 -m unittest discover -s recipe -p 'test_*.py'
PILOT_MACOS_INTEGRATION=1 python3 -m unittest discover -s recipe -p test_pilot_macos.py
```

The opt-in lifecycle cases use harmless short-lived subprocesses and dummy checkpoint files,
not model weights. They exercise real process groups, locks, loopback HTTP and cancellation.
Never run the following model checks without a coordinated compute window.

For a freshly started single-lane server with builtin default and two attested catalog voices:

```sh
python3 bench/reference_cache_test.py --base http://127.0.0.1:18793 \
  --voice synthetic-a --other-voice synthetic-b --out cache-transitions.json
python3 bench/stream_check.py --base http://127.0.0.1:18793 --model fishaudio/drama-3 \
  --voice synthetic-a --input "The train leaves at seven fifteen." --out cached-stream.json
```

The transition check compares actual PCM across same-voice, builtin-default, other-voice and
changed-text requests; it does not accept a log message as correctness proof. The stream check
also abandons a cached-reference stream and verifies that the next reply is unchanged.

Capture the full corpus twice against the uncached baseline, then twice against the candidate,
using the same reference voice and unchanged server settings:

```sh
python3 bench/corpus_capture.py http://127.0.0.1:18793 base-capture --passes 2 --voice synthetic-a
# Stop and settle the baseline; start the candidate in a fresh coordinated window.
python3 bench/corpus_capture.py http://127.0.0.1:18793 new-capture --passes 2 --voice synthetic-a
python3 bench/compare_gate.py base-capture base-capture new-capture gate.json
```

All requests must succeed and the complete corpus must be present before interpreting the gate.
Do not turn a correctness run, component timing, or cross-host comparison into a speed claim.

### Recorded repair verification

[`results/macos-pr17-repair.json`](../results/macos-pr17-repair.json) records the corrected
patch-0012 engine `c265ca2fc4b9b54ba75b6780e110c566483c93d2`, its binary hashes, CPU checks and
the full-model live comparison against the 11-patch uncached baseline. All 48 paired corpus
captures matched; voice/default/text transitions, builtin and cached streaming abandonment,
reference-boundary variants and two-lane parity passed. Eight warmed voice switches showed
0 KiB process-RSS spread on each build. RSS is not direct Metal allocation telemetry.
All three servers and their owned helpers settled.
This correctness evidence makes no speed, CUDA-runtime or NVFP4-Mac claim.
