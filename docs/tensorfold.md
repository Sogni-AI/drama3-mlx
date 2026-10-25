# TensorFold version

The native version is the original
**[drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold)** implementation by
**[netrunner (@plotarmordev)](https://github.com/plotarmordev)** and its contributors.
It lives in [`tensorfold/`](../tensorfold/), alongside the MLX Python package.
Built with Fish Audio.

## Get and run it

```sh
git clone https://github.com/Sogni-AI/drama3-mlx.git
cd drama3-mlx/tensorfold
```

An ordinary clone includes the TensorFold recipe; no submodule initialization is needed.
Run its commands from this `tensorfold/` directory so its relative paths resolve correctly.

| Platform | Instructions | Launcher |
|---|---|---|
| Apple Silicon / native Metal | [macOS setup, build, voices and admission rules](../tensorfold/recipe/README-macos.md) | `python3 recipe/pilot_macos.py` |
| Linux / native CUDA on GB10 | [CUDA prerequisites and quickstart](../tensorfold/README.md#quickstart) | `python3 recipe/pilot.py` |

Both use the recipe's pinned engine and checkpoint revisions. Follow its instructions for
Zig 0.17.0, building with `--weights nvfp4` or `--weights full`, checkpoint verification,
the shared GPU lock and available memory. The native engine is built separately from the
MLX package and serves `/v1/audio/speech`; `pip install .` installs the MLX CLI and Python API.

The original [recipe](../tensorfold/recipe/README.md), [benchmarks](../tensorfold/bench/README.md),
[qualification tools](../tensorfold/qualification/README.md) and
[results](../tensorfold/results/README.md) are included with their original context.
Published measurements describe the upstream runs, not a new measurement of this combined repository.

## Source and attribution

| Item | Imported source |
|---|---|
| Repository | [plotarmordev/drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold) |
| Revision | [`a7752b4591da9b33bf6e7f8f2b478a638bc64bd1`](https://github.com/plotarmordev/drama-tensorfold/commit/a7752b4591da9b33bf6e7f8f2b478a638bc64bd1), upstream `main` when imported |
| Location | [`tensorfold/`](../tensorfold/), unchanged from that revision |
| Code license and notices | [Apache-2.0 LICENSE](../tensorfold/LICENSE), [original NOTICE](../tensorfold/NOTICE) |
| Full-model engine | [`engine/UPSTREAM`](../tensorfold/engine/UPSTREAM) and [patches](../tensorfold/engine/patches/README.md) |
| CUDA NVFP4 engine | [`engine/nvfp4/UPSTREAM`](../tensorfold/engine/nvfp4/UPSTREAM) |
| Metal NVFP4 engine | [`engine/nvfp4/macos/UPSTREAM`](../tensorfold/engine/nvfp4/macos/UPSTREAM) and its patches |

The merge preserves upstream Git history and author identities. Additional credit goes to
[@jerryfane](https://github.com/jerryfane), [@BobClawblaw](https://github.com/BobClawblaw),
the [TensorFold contributors](https://github.com/ashhart/TensorFold) and Fish Audio.
See the repository-wide [Credits](../CREDITS.md) and [NOTICE](../NOTICE).

The imported source is upstream's merged `main` branch at the revision above. The separate proposed
changes for [protected/CJK chunking](https://github.com/plotarmordev/drama-tensorfold/pull/26),
[request sampling](https://github.com/plotarmordev/drama-tensorfold/pull/27) and
[dialogue speakers](https://github.com/plotarmordev/drama-tensorfold/pull/28) are outside this
pinned snapshot. The MLX implementation already supplies those underlying capabilities;
the native HTTP and Python APIs have their own documented contracts.

## Maintaining the import

This is a Git subtree with preserved history. After reviewing a newer upstream revision,
update from the root of the combined repository:

```sh
git subtree pull --prefix=tensorfold https://github.com/plotarmordev/drama-tensorfold.git main
```

Then update the revision above, retain upstream licenses and notices, inspect the complete
diff and run the recipe tests and checksums. Keep Sogni integration notes outside `tensorfold/`
so an unmodified import can be compared directly with its upstream tree. Any future edits
to imported files must be identified as modifications and preserve the original attribution.

The MLX code and model weights retain their [Fish Audio license](../LICENSE).
The separate TensorFold code license does not change the model's license or usage terms.
