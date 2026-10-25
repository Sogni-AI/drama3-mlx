#!/usr/bin/env python3
"""Derives the revised worst-row cosine bar from reference data alone (CPU only, numpy only).

The bar is the lowest minimum row cosine, over the eight co-batched refself views, of each view's final-norm hidden
against the solo-first view. A candidate passes iff candidate >= bar, compared at binary64 with no rounding or
margin. The solo views are determinism controls, not bar views: both must be byte-identical to the frozen reference.
Native data is never read.

Usage: worst-row-bar.py REFSELF_OUT BASELINE
  REFSELF_OUT  refself.py output directory, finalized (final.json evidence_valid)
  BASELINE     the frozen backbone-reference-check.py output it was scored against
"""
import json
from pathlib import Path
import sys

import numpy as np

VIEWS = ('companion-2-before', 'companion-7-before', 'companion-128-before', 'companion-2-after',
         'companion-7-after', 'companion-128-after', 'pair-0', 'pair-1')
REGISTERED_BAR = 0.8712518909203973
HIDDEN = 2048


def load(path: Path, rows: int) -> np.ndarray:
    bits = np.fromfile(path, dtype='<u2').reshape(rows, HIDDEN)
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def row_cosines(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dots = np.sum(candidate * reference, axis=1)
    norms = np.linalg.norm(candidate, axis=1) * np.linalg.norm(reference, axis=1)
    return np.where(norms > 0, dots / np.where(norms > 0, norms, 1), 1.0)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run, baseline = Path(sys.argv[1]), Path(sys.argv[2])
    if not json.loads((run / 'final.json').read_text())['evidence_valid']:
        raise SystemExit(f'{run / "final.json"} is not valid evidence')
    frozen = (baseline / 'reference-hidden.bf16').read_bytes()
    if not (run / 'solo-first-hidden.bf16').read_bytes() == (run / 'solo-last-hidden.bf16').read_bytes() == frozen:
        raise SystemExit('the solo controls are not byte-identical to the frozen reference')
    rows = json.loads((baseline / 'fixture.json').read_text())['tokens']
    reference = load(run / 'solo-first-hidden.bf16', rows)
    views = {}
    for view in VIEWS:
        c = row_cosines(load(run / f'{view}-hidden.bf16', rows), reference)
        views[view] = {'min_cosine': float(c.min()), 'min_row': int(c.argmin()),
                       'rows_below': {str(t): int(np.count_nonzero(c < t)) for t in (0.995, 0.99, 0.95)},
                       'quantiles': dict(zip(('q0', 'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'q100'),
                                             np.quantile(c, [0, .01, .05, .25, .5, .75, .95, .99, 1],
                                                         method='linear').tolist()))}
    bar = min(v['min_cosine'] for v in views.values())
    print(json.dumps({'views': views, 'bar': bar, 'registered_bar': REGISTERED_BAR,
                      'reproduces_registered_bar': bar == REGISTERED_BAR}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
