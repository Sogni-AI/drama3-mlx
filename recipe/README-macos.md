# macOS: running this recipe on Apple silicon

The recipe is written for a GB10 host (Linux, user systemd, CUDA 13, `nvcc`). TensorFold itself
builds Drama 3 for Metal as well, so the same patched engine serves `fishaudio/drama-3` on a Mac
with three changes: no `nvcc`, no `systemd-run`, and no `/proc/meminfo`. Everything else — the
pinned engine commit, the patch series, the checkpoint revision and its checksums, the loopback
port, the HTTP contract — is unchanged.

Verified on: Mac Studio, Apple M5 Ultra, 256 GiB unified memory, macOS 27.0.1, Zig 0.17.0,
engine `d1018b318fb51981c4a18f12e40269a4d3ee9713` + all 11 patches = `881151ca`.

## 1. Prerequisites

- Zig 0.17.0 (`https://ziglang.org/download/0.17.0/zig-aarch64-macos-0.17.0.tar.xz`,
  sha256 `b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a`).
- Xcode command line tools (the build shells out to `xcrun metal`).
- ~39 GiB for the checkpoint, a few GiB for the build tree and caches.
- No CUDA toolkit. `scripts/build-engine.sh` detects Darwin and drops `-Dnvcc`, because the
  CUDA option is only declared for Linux/Windows targets.

## 2. Checkpoint

```
hf download fishaudio/drama-3 --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 --local-dir drama-3
(cd drama-3 && shasum -a 256 -c ../recipe/checksums/checkpoint.sha256)
```

All ten files must print `OK`. `shasum` replaces `sha256sum`, which macOS does not ship.

## 3. Build

```
ZIG=/path/to/zig-0.17.0/zig scripts/build-engine.sh
```

Output: `build/tensorfold/zig-out/native/bin/tensorfold-native` and
`build/tensorfold/zig-out/bin/drama3-generate`. The Metal path is selected by the target OS;
nothing else changes.

## 4. Launch

`recipe/pilot.py` needs Linux and user systemd, so macOS uses `recipe/pilot_macos.py`. It keeps
the recipe's guard and evidence layout — shared GPU lock, admission check, port check,
`DRAMA_PILOT_READY`, `session.json` / `ready.json` / `memory.jsonl` / `server.log` /
`completion.json` — and replaces the Linux-only parts:

| Linux | macOS replacement |
|---|---|
| `systemd-run --user` unit, `MemoryMax=54` | direct child in its own process group, stopped by signal |
| `/proc/meminfo` `MemAvailable` | `vm_stat` free + inactive + speculative + purgeable |
| `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54` cap | none (the Metal path has no such cap) |
| admission: `max(weights+30, 54+30+4)` | admission: `weights + 30` GiB, or `--min-available-gib` |

```
touch runtime/gpu.lock          # this host has no pre-existing shared GPU lock
python3 recipe/pilot_macos.py \
  --native-bin build/tensorfold/zig-out/native/bin/tensorfold-native \
  --checkpoint drama-3 --runtime-dir runtime --gpu-lock runtime/gpu.lock \
  --evidence-dir runs/pilot-01 [--min-available-gib 60] [-- --max-frames 4096]
```

The launcher binds `--host 0.0.0.0` by default (pass `--host 127.0.0.1` for loopback only). The
recipe's own advice is loopback; on a trusted LAN, exposing it makes the server reachable from
other hosts. The Fish Audio licence's Hosted Service clause (III(c)) is about third parties
sending their own text, which a home LAN does not really do, but read Section III before
exposing it beyond your own machines.

### As a supervised service

`~/Library/LaunchAgents/com.tensorfold.drama3.plist` runs the launcher on port 21793 with
`RunAtLoad`, `KeepAlive` and a 60 s throttle, logging to `~/Library/Logs/drama3.log`. A killed
server is recorded as a failure by the launcher, which exits; launchd then re-runs it, and the
admission check refuses to load again until memory allows it.

```
launchctl kickstart -k gui/$(id -u)/com.tensorfold.drama3    # restart
launchctl bootout gui/$(id -u)/com.tensorfold.drama3         # stop
```

On this host another model (Ornith-1.5-122B) holds ~67 GiB of wired memory, so `vm_stat`
reports ~66 GiB available and the default `weights + 30` floor (68.6 GiB) refuses to start.
`--min-available-gib 60` was used deliberately; the figure is recorded in `session.json` and the
34 GiB abort threshold still applied. With the other model stopped, the default floor admits.

## 5. Synthesize

```
curl -sS -D headers.txt http://127.0.0.1:18793/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "The train leaves at seven fifteen."}' -o hello.wav
```

`aplay` does not exist on macOS; use `afplay hello.wav`, or write the stream to a file:

```
curl -sSN http://127.0.0.1:18793/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"fishaudio/drama-3","input":"The train leaves at seven fifteen.","response_format":"pcm","stream":true}' \
  -o hello.pcm     # mono s16le 44.1 kHz; afplay needs a WAV header, so convert or use the wav route
```

## 6. Cloning a voice

The route takes no reference audio per request; only catalog names select a voice, so cloning is
a build step, not a request field. The catalog format is `drama3-http-voice/1` with a `source`
attestation (`synthetic`, `self` or `consented`) and up to four `{text, codes}` references, where
`codes` is the encoded VQ rows of a reference clip. Each catalog file must stay under 256 KB and
the server loads it with `--voices DIR`; `GET /v1/voices` lists what loaded.

```
# 1. a clean clip of the target speaker, ~8-10 s, one voice only, 44.1 kHz mono
ffmpeg -ss 45.88 -to 54.10 -i source.webm -ac 1 -ar 44100 -c:a pcm_s16le clip.wav
# 2. the transcript must match the clip; it is the conditioning text
whisper-cli -m ~/models/whisper/ggml-small.en.bin -f clip.wav -nt
# 3. encode the clip to VQ rows (codec only, no backbone load)
build/tensorfold/zig-out/bin/drama3-generate --codec drama-3/codec.pth \
  --encode clip.codes --ref-audio clip.wav
# 4. build the catalog into runtime/voices/<name>.json
python3 recipe/build_voice.py NAME "note" "TRANSCRIPT:clip.codes"
```

Then serve with `-- --voices /abs/path/to/runtime/voices` (absolute: the launcher runs with the
runtime directory as cwd) and request `{"voice":"NAME"}`. Without `--default-voice`, `default`
stays the built-in voice. On this host the service runs `--default-voice jones`, so a request
that omits `voice` (or sends `default`) is spoken in the Jones clone; the built-in voice is then
reachable only by adding another catalog name.

Measured here (2026-10-21): Morgan Freeman and James Earl Jones catalogs. Freeman uses an 8.2 s
interview clip; Jones uses a 10.3 s silence-trimmed cut of his 1993 Fresh Air interview (the
14.3 s source carried 6.5 s of pauses, so trimming cut prefill without losing any speech). Same
sentence across voices: default, freeman and jones all returned complete audio with
`finish=stop`, and Whisper transcribed each back correctly.

Latency is dominated by reference prefill, not decoding. Measured on this host: decode runs at
about 0.46x realtime, while time to first audio scales with the reference at roughly 3 s per 220
frames — 0.40 s for the built-in voice, 3.26 s for the 10.3 s Jones reference, 4.57 s for the
untrimmed 14.3 s source, 7.90 s for a 27.4 s source. Drama refuses cache modifiers ("needs
context 1025..8192, 1..8 frame lanes and no text, vision, TP or cache modifiers"), so
`--prompt-cache-gib` cannot be used; `--keep-warm` made no measurable difference; `--parallel 4`
stops overlapping requests from being refused (three concurrent calls all returned 200 instead of
503) but they serialize on the GPU, so it buys throughput, not single-request latency.

The series here carries `engine/patches/0012`, which caches the reference prefill in the Metal
host: the first request for a voice prefills the reference and snapshots the recurrent state, and
later requests with the same reference restore it and prefill only their own text. Measured after
the patch, same service: first request 3.59 s to first audio, subsequent ones 1.37-1.41 s, and the
replies are byte-identical to the uncached engine (same SHA-256 for the same prompt and seed). The
snapshot is about 66 MB per lane and is rebuilt when the reference rows change, so a different
voice pays one uncached request. The full Genesis paragraph went from 4.56 s to 2.27 s to first
audio, 20.63 s to 18.32 s total, with identical audio.

Lessons from building them:

- Reference noise carries into the output. A 1970s TV interview reference (-42.6 dBFS quiet
  floor) cloned to -49.3 dBFS; the Fresh Air reference (-57.1 dBFS) cloned to -56.3 dBFS.
  Measure the quietest-10% RMS of a candidate clip before encoding it; anything above about
  -50 dBFS is too noisy for a clean clone. Studio narration and NPR interviews are the clean
  sources; film dialogue carries a music bed (the Vader attempt measured -30 dBFS and was
  rejected on sound, not on noise alone).
- Trim silence before encoding. Removing pauses from a reference keeps every spoken word and
  cuts prefill proportionally; on the Jones clip that was 14.3 s down to 10.3 s.
- One short, clean, single-speaker reference is the reliable shape. A two-reference catalog
  (10.6 s + 16.4 s) made the engine return `500 ... finished without complete audio` on one
  text, reproducibly; the single-reference rebuild served the same text fine.
- Cloned voices stall more than the default voice on multi-sentence input. With no guard, a
  three-sentence text produced "This is CNN." followed by 18-27 s of near-silence before EOS.
  `--silence-floor-dbfs -40 --silence-seconds 4 --silence-redraws 1 --silence-redraw-budget-s 60`
  bounds that: the reply ends with `X-Finish-Reason: silence` after 4 s, and the redraw recovered
  the full sentence in most cases. The service runs with those flags.
- Cloning costs latency: measured wall time per request was 3.6-6.1 s for freeman and 4.5-6.3 s
  for jones on 34-79 character texts, versus ~3 s for the default voice. The prompt carries
  hundreds of frames of reference, so prefill dominates.
- The reference clips and the `.codes` files are local only (the repo's CONTRIBUTING keeps audio
  and voice files out of git); only the derived JSON catalogs are needed to serve.


## Measured on this host (2026-10-21)

Same server flags as the recipe (`--context 8192 --parallel 1 --no-drafts
--prompt-cache-gib 0 --keep-warm 0`), three English texts from `bench/corpus`, one request each:

| | result |
|---|---|
| load to `DRAMA_PILOT_READY` | 4.73 s from launcher start to `GET /health` 200 |
| WAV, "The train leaves at seven fifteen." | 2.69 s audio, 237 612 B, `X-Finish-Reason: stop` |
| Whisper `base.en` on that WAV | "The train leaves at 7.15." |
| RTF (39 / 38 / 19 chars) | 0.638 / 0.596 / 0.625 |
| first streamed PCM | 0.328 s |
| peak RSS while serving | ~14.5 GiB resident; 32.2 GB of FP8 experts read in place |
| minimum available memory during the run | 37.6 GiB |
| shutdown | clean, lock released, `completion.json` failures `[]` |

The GB10 reference in the README reports RTF 0.634 and 0.141 s to first audio. RTF matches; first
audio is slower here, and the two numbers come from different hosts, so they are not a
like-for-like comparison.

## Not available on this commit

`--long-form` / `--long-form-chars` are described in the top-level README but are not in the
pinned patch series; `tensorfold-native serve --help` does not list them. `--max-frames`,
`--voices` and `--default-voice` are present and parse. The README's long-reply advice therefore
applies here only through `--max-frames` (up to 8190).

`--backend sglang-omni` is CUDA-only; `pilot_macos.py` refuses it.
