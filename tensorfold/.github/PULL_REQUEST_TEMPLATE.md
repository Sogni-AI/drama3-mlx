## Description and motivation

<!--

    What changes and why. Name the engine patches this adds or changes. If this
    pull request is stacked on another, say which merges first.

-->

---

## Testing

<!--

    The checks you ran and their output: `zig build test`, the voice tool tests,
    `bench/stream_check.py --self-test`, the audio-identity gate, the
    qualification gates. Name any you didn't run and why.

-->

---

## Performance or quality claim

<!--

    Leave "None." unless this pull request claims a speed or quality
    improvement. If it does, follow CONTRIBUTING.md#performance-and-quality-claims
    and replace "None." with the evidence.

-->

None.

---

## Checklist

<!--

    Mark an item that doesn't apply as "n/a", for example the engine items on a
    docs-only change.

-->

- [ ] One change per patch, new patches at the end of the series, `SHA256SUMS` and `engine/patches/README.md`
      updated
- [ ] `scripts/build-engine.sh` applies the whole series and builds from a clean checkout
- [ ] Every behaviour change has a deterministic test that fails without it
- [ ] No dead code, debug output or leftovers
- [ ] No personal voices, recordings, private names, host names or private paths in code, fixtures, docs, evidence
      or commit messages
- [ ] Where audio should not change, `compare_gate.py` reports `"identical": true` and `stream_check.py` passes;
      where it changes on purpose, the qualification gates are reported before and after

### Evidence (only with a claim)

- [ ] Claimed metric and statistic, and the quality margins when audio changes, stated before measuring
- [ ] Hardware, OS, driver, CUDA and Zig versions, checkpoint revision, recipe commits and engine HEADs stated
- [ ] Runs A, B and A2 against the recipe baseline under neutral paths, each from a fresh `recipe/pilot.py` start,
      with the exact commands
- [ ] First-audio time, total time and RTF as warm all-attempts p50 and p95, B against A and A2, with each run's
      failures and `server_process_unchanged`
- [ ] For a speed claim, Grill v0.11.0 speech captures of A, B and A2 and `grill-perf media compare` reporting
      `PASS` on first-audio time (and total time when claimed) with the duration-ratio gate, the compare JSON and
      the three capture directories attached
- [ ] `gate-A2.json` identical; WER, speaker cosine and duration ratio where audio changes, with the ASR and
      speaker-embedding models, their revisions and the commands
- [ ] Every attempt archived unedited, failed ones included, and the privacy check prints nothing; attempts withheld
      for privacy: N, each re-measured in full
