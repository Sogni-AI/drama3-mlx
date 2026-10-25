# Before/after benchmark 02 (speed only)

The sglang-omni reference `ab095ef9` (before) against TensorFold native with the full Drama 3 series (after), on
one GB10 host, one engine at a time under `recipe/pilot.py`. This is a speed comparison, not output equivalence:
the engines' audio differs numerically and in length, so RTF is the duration-normalized view.

- **Workload.** The 24-item corpus at concurrency 1. Whole replies on a fresh process: 1 cold, 2 warmup, 3 warm
  passes. Streamed replies on the same process: 2 warmup, 3 warm passes. Driver: `bench/before-after.sh`.
- **Tools.** `bench/drama_bench.py` `bdf27871…`, `bench/tables.py`, `bench/admit.py`.
- **Caches.** The runtime directory, including its compilation caches, was reused: a fresh process is not a cold
  Triton/JIT cache, and the cold starts are single descriptive starts.
- **Native binary.** `tensorfold-native` `aa211423…`.

[`tables.json`](tables.json) holds every number below (`tables.py` output, SHA-256 `4f295428…`), except the cold
starts, which are in [`startup.json`](startup.json) (each pilot's `ready.json`).

## Results (warm, all attempts, failures in every denominator)

| Measure | Reference | TensorFold native |
|---|---:|---:|
| Whole reply p50 | 7.93 s | 4.89 s (1.62x) |
| RTF p50 (lower is faster) | 0.939 | 0.634 |
| RTF p99 | 1.011 | 0.712 |
| Time to first audio p50 / p99 (stream) | 0.521 / 0.550 s | 0.141 / 0.179 s |
| Cold start (process start to ready) | 254.1 s | 6.4 s |
| Whole / stream failures | 0 / 0 | 0 / 0 |
| 1024-frame replies (whole, warm) | 6 (medium-05 x2, long-02, long-04, long-07 x2) | 3 (long-05 x3) |

| Bucket | Reference p50 | Native p50 | Factor | Reference RTF | Native RTF |
|---|---:|---:|---:|---:|---:|
| short | 2.50 s | 1.86 s | 1.35x | 0.950 | 0.657 |
| medium | 7.93 s | 4.89 s | 1.62x | 0.940 | 0.633 |
| long | 20.22 s | 12.90 s | 1.57x | 0.925 | 0.627 |

- **Cutoff-free cohort.** Excluding medium-05, long-02, long-04, long-05 and long-07 leaves 19 items and 57
  attempts per engine. The p50 factor is 1.708x; the summed-time factor is 1.608 = 1.093 (audio ratio) x 1.47
  (RTF ratio); per item it ranges from 1.217 to 1.902 (median 1.576). Native has the lower per-item median
  latency on all 19 items.
- **All items but long-05.** Native also has the lower per-item median latency on every item except long-05,
  including the items where the reference hit the cap.
- **long-05.** Native generates to the 1024-frame cap on every attempt and returns the clipped audio with
  `X-Finish-Reason: length`. Its 29.66 s p50 covers 47.55 s of audio, against the reference's 23.31 s of audio
  (factor 0.73x).

## Deviations

- **Reference admission.** `admit.py` refused the reference before its load: CUDA free 30.99 GiB and `MemFree`
  31.18 GiB, against 58 GiB required. The pilot guard admitted it on `MemAvailable` (≥ 88 GiB) with no guard
  failures, and the run continued. The reference's cold start and timings therefore ran under different
  page-cache conditions than native's; the effect was not isolated. `bench/before-after.sh` now stops on a
  refused admission.
- **No global page-cache drop** was performed; `admit.py` advises only the run's own files.
