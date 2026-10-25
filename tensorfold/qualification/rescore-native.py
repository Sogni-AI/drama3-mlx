#!/usr/bin/env python3
"""Re-score native backbone outputs under the pre-registered worst-row bar (CPU only).

Metrics come unchanged from backbone-reference-check.py. Only hidden_min_row_cosine is replaced, by the bar that
worst-row-bar.py derives and that was registered before native was re-scored. Every other threshold is the helper's
own. Writes OUT (must not exist).

Usage: rescore-native.py NATIVE_DIR BASELINE OUT
  NATIVE_DIR  the backbone oracle's native/ directory (native-chunk-{1,7,128}-*)
  BASELINE    the frozen backbone-reference-check.py output
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HELPER_PATH = Path(__file__).resolve().with_name('backbone-reference-check.py')
REGISTERED_BAR = 0.8712518909203973
REGISTERED_AT_UTC = '2026-10-16T02:12:59.760971+00:00'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    native, baseline, out = map(Path, sys.argv[1:])
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    spec = importlib.util.spec_from_file_location('helper', HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    fixture = json.loads((baseline / 'fixture.json').read_text())
    rows = fixture['tokens']
    reference_hidden = helper.bf16_bits(baseline / 'reference-hidden.bf16', (rows, helper.HIDDEN))
    reference_semantic = np.fromfile(baseline / 'reference-semantic.f32', dtype='<f4')

    revised = dict(helper.THRESHOLDS, hidden_min_row_cosine=REGISTERED_BAR)
    result = {'schema': 'drama-backbone-rescore/1', 'registered_at_utc': REGISTERED_AT_UTC,
              'thresholds_original': helper.THRESHOLDS, 'thresholds_revised': revised,
              'helper_sha256': sha256(HELPER_PATH), 'script_sha256': sha256(Path(__file__)), 'chunks': {}}
    for chunk in helper.CHUNKS:
        hidden_path = native / f'native-chunk-{chunk}-hidden.bf16'
        semantic_path = native / f'native-chunk-{chunk}-semantic.f32'
        hidden = helper.hidden_metrics(helper.bf16_bits(hidden_path, (rows, helper.HIDDEN)), reference_hidden)
        semantic = helper.semantic_metrics(np.fromfile(semantic_path, dtype='<f4'), reference_semantic, fixture)
        gates = {
            'hidden_min_cosine_all_rows': hidden['all_rows']['cosine'] >= revised['hidden_min_cosine_all_rows'],
            'hidden_max_relative_l2_all_rows': hidden['all_rows']['relative_l2'] <= revised['hidden_max_relative_l2_all_rows'],
            'hidden_min_row_cosine': hidden['rows']['min_cosine'] >= revised['hidden_min_row_cosine'],
            'hidden_min_last_row_cosine': hidden['last_row']['cosine'] >= revised['hidden_min_last_row_cosine'],
            'semantic_top1_equal': semantic['top1_equal'],
            'semantic_min_top30_overlap': semantic['top30_overlap'] >= revised['semantic_min_top30_overlap'],
            'semantic_max_kl_reference_to_native': semantic['kl_reference_to_native'] <= revised['semantic_max_kl_reference_to_native'],
            'semantic_min_cosine': semantic['cosine'] >= revised['semantic_min_cosine'],
            'semantic_max_relative_l2': semantic['relative_l2'] <= revised['semantic_max_relative_l2'],
        }
        result['chunks'][str(chunk)] = {
            'files_sha256': {hidden_path.name: sha256(hidden_path), semantic_path.name: sha256(semantic_path)},
            'hidden_all_rows': hidden['all_rows'], 'hidden_last_row': hidden['last_row'],
            'hidden_min_row_cosine': hidden['rows']['min_cosine'], 'hidden_min_row': hidden['rows']['min_cosine_row'],
            'semantic': {k: semantic[k] for k in ('top1_equal', 'top30_overlap', 'kl_reference_to_native', 'cosine', 'relative_l2')},
            'gates': gates, 'failing_gates': [k for k, v in gates.items() if not v], 'passed': all(gates.values())}
    result['passed'] = all(c['passed'] for c in result['chunks'].values())
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: (v['hidden_min_row_cosine'], v['failing_gates']) for k, v in result['chunks'].items()}))
    print('passed', result['passed'], 'sha256', sha256(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
