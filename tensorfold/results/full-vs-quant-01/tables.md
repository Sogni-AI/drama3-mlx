| Measure | Full FP8 | Quant NVFP4, γ 1.5 | Full FP8, γ 1.5 | Full FP8, built-in voice | Quant NVFP4, built-in voice |
|---|---:|---:|---:|---:|---:|
| Weights + codec on disk (GiB) | 38.63 | 20.76 | 38.63 | 38.63 | 20.76 |
| Peak device memory, /health (GiB) | 36.95 | 19.35 | 37.20 | 36.95 | 19.35 |
| Host MemAvailable drop, max (GiB) | 39.34 | 21.27 | 39.37 | 38.67 | 21.23 |
| Process start to ready, arithmetic median of 4 cold starts (s) | 5.9 | 26.3 | 5.9 | 5.7 | 26.7 |
| Engine load, arithmetic median (s) | 5.7 | 26.1 | 5.8 | 5.6 | 26.5 |
| Real-time factor, warm p50 nearest-rank, salt 0 (all warm passes) | 0.846 | 0.825 | 0.940 | 0.629 | 0.618 |
| Real-time factor, warm p95 nearest-rank, salt 0 (all warm passes) | 1.539 | 1.462 | 1.832 | 0.670 | 0.656 |
| Real-time factor, arithmetic median of each salt's first warm pass, lowest–highest | 0.853–0.866 | 0.824–0.838 | 0.914–0.941 | 0.627–0.640 | 0.611–0.621 |
| Whole reply, warm p50 nearest-rank, salt 0 (s) | 9.53 | 8.38 | 9.30 | 5.33 | 5.37 |
| First audio when streaming, p50 nearest-rank, salt 0 (s) | 2.428 | 1.955 | 2.515 | 0.141 | 0.113 |
| Decode, codec frames/s, streamed, salt 0 (arithmetic median) | 32.9 | 32.0 | 30.3 | 33.8 | 34.1 |
| Prompt tokens per second to first audio (arithmetic median) | 550 | 681 | 530 | 342 | 430 |
| Prompt tokens per request (arithmetic median) | 1335 | 1335 | 1335 | 48 | 48 |
| Warm requests succeeded | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed |
| Word error rate, mean [95 % CI] | 1.42 % [0.00, 3.48] | 1.31 % [0.04, 3.32] | 1.47 % [0.08, 3.46] | 6.23 % [2.45, 10.74] | 2.92 % [0.24, 6.57] |
| Word error rate, pooled | 0.60 % | 0.56 % | 0.71 % | 7.09 % | 1.01 % |
| Word error rate, mean of each salt, lowest–highest | 1.23–1.61 % | 0.91–1.88 % | 1.29–1.69 % | 4.08–11.39 % | 1.80–4.27 % |
| Failed takes (cap, collapsed) | 0/96 (cap 0, collapsed 0) | 0/96 (cap 0, collapsed 0) | 0/96 (cap 0, collapsed 0) | 9/96 (cap 0, collapsed 9) | 5/96 (cap 0, collapsed 5) |
| Failed takes per salt (0 / 1 / 2 / 3) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 / 2 / 1 / 4 | 0 / 2 / 2 / 1 |
| Engine finish reason, scored takes | stop 96 | stop 96 | stop 96 | silence 7, stop 89 | silence 2, stop 94 |
| Speaker cosine to the full model's take | 0.901 (itself, other salt) | 0.900 | 0.899 | 0.309 (itself, other salt) | 0.308 |

Two median conventions. Rows marked nearest-rank are the benchmark client's own summaries (`bench/drama_bench.py`, `percentiles`): the sorted value at rank ⌈p·n/100⌉, never interpolated, so a p50 over an even count is the lower of the two middle values. Rows marked arithmetic median are computed by `bench/full-vs-quant/report.py` with Python's `statistics.median`, which averages the two middle values of an even count; the per-salt real-time-factor range is one of these (its `tables.json` key keeps the name `whole_warm_rtf_p50_per_salt`).

| Paired comparison | Pairs | Failed only in first | Failed only in second | Exact McNemar p | WER difference, mean [95 % CI over items] |
|---|---:|---:|---:|---:|---|
| Quant NVFP4, γ 1.5 vs Full FP8 | 96 | 0 | 0 | 1.000 | -0.11 pp [-0.35, +0.08] |
| Full FP8, γ 1.5 vs Full FP8 | 96 | 0 | 0 | 1.000 | +0.05 pp [-0.24, +0.34] |
| Quant NVFP4, γ 1.5 vs Full FP8, γ 1.5 | 96 | 0 | 0 | 1.000 | -0.16 pp [-0.47, +0.00] |
| Quant NVFP4, built-in voice vs Full FP8, built-in voice | 96 | 3 | 7 | 0.344 | -3.31 pp [-7.88, +0.54] |
