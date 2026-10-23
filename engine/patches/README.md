# Engine patch series

The `*.patch` files here are opt-in patches made with `git format-patch` against the upstream TensorFold commit
pinned in [`../UPSTREAM`](../UPSTREAM). `scripts/build-engine.sh` applies them in file-name order with `git am`.
Each one is a feature branch's net diff over the patches before it; the list, the flag each adds and the upstream
pull request that carries it are in the [README](../../README.md#the-patches).

All patches are Apache-2.0 ([`../../LICENSE`](../../LICENSE)). The Drama 3 model weights and codec checkpoint the
engine loads are under the Fish Audio Community License v1.0 and are downloaded separately; see
[`NOTICE`](../../NOTICE).

`SHA256SUMS` pins every patch.

Patch 0012 is corrected in place: it owns one reusable Metal recurrent-state snapshot per lane,
invalidates incompatible live KV before prefill, and uses the prompt builder's exact invariant
boundary including tower lookahead. It preserves uncached chunk cuts and CUDA execution.
Its CPU regressions cover identity, cancellation/error invalidation, allocation ownership and
prompt boundaries; real-model transition/corpus/stream checks are described in
[`recipe/README-macos.md`](../../recipe/README-macos.md#correctness-gates).
