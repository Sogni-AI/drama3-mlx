# Credits

Built with Fish Audio.

## TensorFold implementation

**[netrunner (@plotarmordev)](https://github.com/plotarmordev)** created
[drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold), the native Drama 3
recipe, engine patches, qualification tools and published benchmark work included in
[`tensorfold/`](tensorfold/). Credit for that work belongs to its original authors.

The import preserves the complete Git ancestry through
[`a7752b4591da9b33bf6e7f8f2b478a638bc64bd1`](https://github.com/plotarmordev/drama-tensorfold/commit/a7752b4591da9b33bf6e7f8f2b478a638bc64bd1),
including contributions by [@jerryfane](https://github.com/jerryfane) and
[@BobClawblaw](https://github.com/BobClawblaw). The
[upstream contributor list](https://github.com/plotarmordev/drama-tensorfold/graphs/contributors)
and original commits retain the detailed authorship record.

The native engine is by the [TensorFold contributors](https://github.com/ashhart/TensorFold).
The recipe pins that engine and [plotarmordev's CUDA fork](https://github.com/plotarmordev/TensorFold).
Their original notices, including earlier MIT and third-party notices, remain in the engine
checkouts produced by the recipe. The recipe's SGLang-Omni reference tooling credits
[SGLang-Omni](https://github.com/sgl-project/sglang-omni).

The imported recipe's [Apache-2.0 LICENSE](tensorfold/LICENSE) and complete
[NOTICE](tensorfold/NOTICE) are preserved verbatim. Its historical benchmark and qualification
results retain their original authorship, hardware, revision and caveats; importing them is
not a new performance or quality measurement by Sogni.

## Models and MLX implementation

- [Fish Audio](https://github.com/fishaudio/fish-speech) provides Drama 3, Fish Speech and
  the codec. The MLX port derives from Fish Speech v3.0.0-beta and retains the
  [Fish Audio Community License](LICENSE) and [NOTICE](NOTICE).
- [plotarmordev](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned) provides the
  pruned NVFP4 derivative checkpoint supported by both implementations. Its own license
  and notices remain with separately downloaded weights.
- [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm)
  and [mlx-audio](https://github.com/Blaizzy/mlx-audio) provide the MLX runtime, backbone
  and codec dependencies under their respective MIT licenses.
- [Sogni-AI/drama3-mlx contributors](https://github.com/Sogni-AI/drama3-mlx) maintain
  this MLX port and the documentation that brings both implementations into this repository.

Project names identify their source and authors. Inclusion here does not imply endorsement
by Fish Audio, plotarmordev, TensorFold or their contributors.
