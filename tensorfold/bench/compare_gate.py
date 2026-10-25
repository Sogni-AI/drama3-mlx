#!/usr/bin/env python3
"""Audio-identity gate for a speed change: normal items must match the baseline capture byte for byte (every pass),
and long-05, which reaches the frame cap, must match the clipped reference capture. Timing compares the last pass of
both runs on normal items. Every directory is a corpus_capture.py output.

Usage: compare_gate.py BASE_CAPTURE_DIR CLIPPED_CAPTURE_DIR NEW_CAPTURE_DIR OUT.json
"""
import json
from pathlib import Path
import sys


def rows(path):
    return {(r['pass'], r['item']): r for r in json.loads(Path(path).read_text())['rows']}


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    base_dir, clipped_dir, new_dir, out = sys.argv[1:]
    base, new = rows(Path(base_dir) / 'capture.json'), rows(Path(new_dir) / 'capture.json')
    clipped = {i: r for (_, i), r in rows(Path(clipped_dir) / 'capture.json').items()}
    passes = sorted({p for p, _ in new})
    normal = sorted({i for _, i in new if i != 'long-05'})
    mismatched = [f'{p}/{i}' for p in passes for i in normal
                  if (new[(p, i)]['status'], new[(p, i)]['sha256']) != (base[(p, i)]['status'], base[(p, i)]['sha256'])]
    long05 = [new[(p, 'long-05')]['sha256'] == clipped['long-05']['sha256'] for p in passes]
    last = passes[-1]
    base_s = sum(base[(last, i)]['seconds'] for i in normal)
    new_s = sum(new[(last, i)]['seconds'] for i in normal)
    audio = sum(base[(last, i)]['audio_s'] for i in normal)
    report = {'identical': not mismatched and all(long05), 'normal_mismatches': mismatched,
              'long05_clipped_identical': long05, 'base_rtf': base_s / audio, 'new_rtf': new_s / audio,
              'speedup_vs_base': base_s / new_s,
              'per_item_speedup': {i: base[(last, i)]['seconds'] / new[(last, i)]['seconds'] for i in normal}}
    with open(out, 'x') as f:
        f.write(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('identical', 'normal_mismatches', 'long05_clipped_identical',
                                             'base_rtf', 'new_rtf', 'speedup_vs_base')}))
    return 0 if report['identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
