# Changelog

## Unreleased

### Fixed

- Metal reference-prefill reuse now invalidates before incompatible/default prefill, compares the
  complete invariant prefix plus tower lookahead, preserves original chunk boundaries, and reuses
  one GPU-owned snapshot buffer pair per lane instead of allocating on every voice switch.
- The macOS pilot defaults to loopback, uses one conservative free+inactive memory snapshot,
  validates admission controls, clears reused readiness under the shared lock, and settles its
  owned process group before reaping its leader. The server inherits the lock; unexpected clean
  exits and observer/cleanup failures are reported as failures, not successful completion.
- Voice catalog creation requires explicit synthetic/self/consented provenance and enforces the
  pinned server's name, UTF-8, transcript, reference-count, codec-row and size limits.
- Reference-cache regression checks now compare PCM across voice/default/text transitions.
  Corpus and streaming identity checks accept an explicit catalog voice.


### Added

- `--weights nvfp4|full` on `scripts/build-engine.sh` and `recipe/pilot.py`: the NVFP4 pruned weights serve with
  the same two commands as the full model. The build checks out the engine pinned in `engine/nvfp4/UPSTREAM`; the
  launcher downloads the pinned checkpoint on first use, checks it against `recipe/checksums/nvfp4.sha256` on every
  start, refusing missing or changed files before it takes the GPU lock, and serves it with
  `--cfg-gamma 1.5 --cfg-batched` unless a guidance flag of your own follows `--`. Without `--weights`, both behave
  as before.
- Every speed or quality claim in a pull request carries before/after evidence from the recipe's bench and
  qualification scripts, measured against the recipe baseline (`CONTRIBUTING.md`, pull request template).
- A speed claim also carries Grill v0.11.0 speech evidence: A, B and A2 captures of the recipe server and a
  `grill-perf media compare` that passes on first-audio time and duration ratio (`CONTRIBUTING.md`, pull request
  template). Quality claims keep the WER and speaker-cosine evidence, which the Grill doesn't measure yet.
