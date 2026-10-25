#!/usr/bin/env python3
"""Before/after tables from four drama_bench runs over the same corpus and schedule: reference and native, each
whole (non-streaming) and streamed. Statistics come only from the warm phase. Every displayed percentile, RTF
included, ranks all attempts with failures ordered after successes (it reads "failure" when it lands on one);
per-request RTF is total seconds / audio seconds; CV and audio-duration medians are over successes. A cutoff-free
cohort (items with no whole-reply failure or 1024-frame reply on either engine) is reported beside the all-attempt
tables, never instead of them. Writes tables.json and prints Markdown. A SPEED comparison: the engines' audio is
not byte-comparable."""
import argparse
import json
import math
import statistics
from pathlib import Path

BUCKETS = ('short', 'medium', 'long')


def samples(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / 'samples.jsonl').read_text().splitlines() if line]


def warm(run: Path) -> dict[str, list[dict]]:
    by_item: dict[str, list[dict]] = {}
    for s in samples(run):
        if s['phase'] == 'warm':
            by_item.setdefault(s['item'], []).append(s)
    return by_item


def rank(values: list[float], failed: int, p: int):
    ordered, total = sorted(values), len(values) + failed
    r = max(1, -(-p * total // 100))
    return ordered[r - 1] if r <= len(ordered) else 'failure'


def dist(rows: list[dict], key: str) -> dict:
    ok = [s[key] for s in rows if s['outcome'] == 'ok']
    failed = len(rows) - len(ok)
    mean = statistics.fmean(ok) if ok else None
    cv = statistics.stdev(ok) / mean if len(ok) > 1 and mean else None
    return {'n': len(rows), 'failed': failed, 'p50': rank(ok, failed, 50), 'p99': rank(ok, failed, 99), 'cv': cv}


def audio(rows: list[dict]) -> float | None:
    seconds = [s['audio']['duration_s'] for s in rows if s['outcome'] == 'ok']
    return statistics.median(seconds) if seconds else None


def failures(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for s in rows:
        if s['outcome'] != 'ok':
            out[s['outcome']] = out.get(s['outcome'], 0) + 1
    return out


def ratio(a, b):
    return a / b if isinstance(a, float) and isinstance(b, float) and b else None


FRAME_LIMIT = 1024  # the Drama 3 contract on both engines (drama_host.zig max_frames; yaml max_new_tokens)


def limit_hits(rows: list[dict]) -> int:
    """Successful replies that carry exactly the frame limit: the engine stopped at the cap, not at end of speech."""
    return sum(1 for s in rows if s['outcome'] == 'ok' and s['audio']['frames'] >= FRAME_LIMIT)


def row(ref_whole, nat_whole, ref_stream, nat_stream, ref_streams: bool) -> dict:
    ref_t, nat_t = dist(ref_whole, 'total_s'), dist(nat_whole, 'total_s')
    return {
        'reference_total_s': ref_t, 'native_total_s': nat_t,
        'improvement_p50': ratio(ref_t['p50'], nat_t['p50']),
        'native_first_audio_s': dist(nat_stream, 'first_frame_s') if nat_stream else None,
        'reference_first_audio_s': dist(ref_stream, 'first_frame_s') if ref_stream and ref_streams else None,
        'reference_audio_s_p50': audio(ref_whole), 'native_audio_s_p50': audio(nat_whole),
        'reference_rtf': dist(ref_whole, 'rtf'), 'native_rtf': dist(nat_whole, 'rtf'),
        'reference_failures': failures(ref_whole), 'native_failures': failures(nat_whole),
        'reference_stream_failures': failures(ref_stream or []), 'native_stream_failures': failures(nat_stream or []),
        'reference_frame_limit_replies': limit_hits(ref_whole), 'native_frame_limit_replies': limit_hits(nat_whole),
    }


def fmt(value, digits=2):
    if value is None:
        return 'N/A'
    if value == 'failure':
        return 'FAIL'
    return f'{value:.{digits}f}'


def fails(*groups: dict) -> str:
    parts = [', '.join(f'{k} {v}' for k, v in g.items()) or '0' for g in groups]
    return ' / '.join(parts)


def markdown(title: str, rows: dict[str, dict]) -> str:
    lines = [f'### {title}', '',
             '| | ref p50 s | native p50 s | factor | native TTFA p50 s | ref TTFA p50 s | ref audio s | native audio s '
             '| ref RTF p50 (all attempts) | native RTF p50 (all attempts) | ref p99 s | native p99 s | ref CV | native CV '
             '| failures ref / native '
             '(whole) | stream failures ref / native | 1024-frame replies ref / native |',
             '|' + '---|' * 17]
    for name, r in rows.items():
        nat_ttfa = r['native_first_audio_s']['p50'] if r['native_first_audio_s'] else None
        ref_ttfa = r['reference_first_audio_s']['p50'] if r['reference_first_audio_s'] else None
        lines.append(f"| {name} | {fmt(r['reference_total_s']['p50'])} | {fmt(r['native_total_s']['p50'])} "
                     f"| {fmt(r['improvement_p50'])}x | {fmt(nat_ttfa, 3)} | {fmt(ref_ttfa, 3)} "
                     f"| {fmt(r['reference_audio_s_p50'])} | {fmt(r['native_audio_s_p50'])} "
                     f"| {fmt(r['reference_rtf']['p50'], 3)} | {fmt(r['native_rtf']['p50'], 3)} "
                     f"| {fmt(r['reference_total_s']['p99'])} | {fmt(r['native_total_s']['p99'])} "
                     f"| {fmt(r['reference_total_s']['cv'], 3)} | {fmt(r['native_total_s']['cv'], 3)} "
                     f"| {fails(r['reference_failures'], r['native_failures'])} "
                     f"| {fails(r['reference_stream_failures'], r['native_stream_failures'])} "
                     f"| {r['reference_frame_limit_replies']} / {r['native_frame_limit_replies']} |")
    return '\n'.join(lines)


def summary3(values: list[float]) -> dict:
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def pooled(runs: dict, members: list[str]) -> dict:
    """Whole replies of ``members`` on both engines: all-attempt p50s and the exact aggregate decomposition
    reference time / native time = (reference audio / native audio) x (reference RTF / native RTF), RTF = time / audio."""
    rw = [s for i in members for s in runs['rw'][i]]
    nw = [s for i in members for s in runs['nw'][i]]
    r = row(rw, nw, None, None, False)
    time = {'reference': sum(s['total_s'] for s in rw), 'native': sum(s['total_s'] for s in nw)}
    audio = {'reference': sum(s['audio']['duration_s'] for s in rw), 'native': sum(s['audio']['duration_s'] for s in nw)}
    rtf = {k: time[k] / audio[k] for k in time}
    return {'items': members, 'attempts_per_engine': len(rw),
            'reference_total_s': r['reference_total_s'], 'native_total_s': r['native_total_s'],
            'factor_p50': r['improvement_p50'], 'reference_rtf': r['reference_rtf'], 'native_rtf': r['native_rtf'],
            'sum_time_s': time, 'sum_audio_s': audio, 'aggregate_rtf': rtf,
            'time_factor': time['reference'] / time['native'], 'audio_ratio': audio['reference'] / audio['native'],
            'rtf_ratio': rtf['reference'] / rtf['native']}


def cutoff_free(runs: dict, items: list[str]) -> dict:
    """Items whose whole replies all succeeded below the frame limit on both engines, as a conditional sensitivity
    view; plus per-item ratios over those items and a post hoc length-matched subset (durations within 5%)."""
    clean = [i for i in items if all(s['outcome'] == 'ok' and s['audio']['frames'] < FRAME_LIMIT
                                     for key in ('rw', 'nw') for s in runs[key][i])]
    per_item = {}
    for i in clean:
        median = lambda key, field: statistics.median(s[field] if field != 'audio' else s['audio']['duration_s']
                                                      for s in runs[key][i])
        per_item[i] = {'factor_p50': rank([s['total_s'] for s in runs['rw'][i]], 0, 50) /
                       rank([s['total_s'] for s in runs['nw'][i]], 0, 50),
                       'audio_ratio': median('rw', 'audio') / median('nw', 'audio'),
                       'rtf_ratio': median('rw', 'rtf') / median('nw', 'rtf')}
    matched = [i for i in clean if 0.95 <= 1 / per_item[i]['audio_ratio'] <= 1.05]
    return {'rule': 'drop an item from both engines if either engine has any warm whole-reply failure or 1024-frame reply',
            'excluded': [i for i in items if i not in clean], 'all_items': pooled(runs, clean),
            'buckets': {b: pooled(runs, [i for i in clean if i.startswith(b + '-')]) for b in BUCKETS},
            'per_item': per_item,
            'per_item_summary': {k: summary3([v[k] for v in per_item.values()]) for k in ('factor_p50', 'audio_ratio', 'rtf_ratio')},
            'length_matched_5pct_post_hoc': pooled(runs, matched)}


def cohort_markdown(view: dict) -> str:
    lines = ['### Cutoff-free cohort (sensitivity view; excludes ' + ', '.join(view['excluded']) + ')', '',
             '| | items | attempts/engine | ref p50 s | native p50 s | factor p50 | ref RTF p50 | native RTF p50 '
             '| ref sum s / audio s | native sum s / audio s | time factor = audio ratio x RTF ratio |',
             '|' + '---|' * 11]
    for name, p in {**view['buckets'], 'all retained': view['all_items'],
                    'length-matched within 5% (post hoc)': view['length_matched_5pct_post_hoc']}.items():
        lines.append(f"| {name} | {len(p['items'])} | {p['attempts_per_engine']} | {fmt(p['reference_total_s']['p50'], 3)} "
                     f"| {fmt(p['native_total_s']['p50'], 3)} | {fmt(p['factor_p50'], 3)}x "
                     f"| {fmt(p['reference_rtf']['p50'], 3)} | {fmt(p['native_rtf']['p50'], 3)} "
                     f"| {fmt(p['sum_time_s']['reference'], 1)} / {fmt(p['sum_audio_s']['reference'], 1)} "
                     f"| {fmt(p['sum_time_s']['native'], 1)} / {fmt(p['sum_audio_s']['native'], 1)} "
                     f"| {fmt(p['time_factor'], 3)} = {fmt(p['audio_ratio'], 3)} x {fmt(p['rtf_ratio'], 3)} |")
    s = view['per_item_summary']
    lines += ['', 'Per retained item (min / median / max): factor p50 '
              f"{fmt(s['factor_p50']['min'], 3)} / {fmt(s['factor_p50']['median'], 3)} / {fmt(s['factor_p50']['max'], 3)}; "
              f"reference/native audio {fmt(s['audio_ratio']['min'], 3)} / {fmt(s['audio_ratio']['median'], 3)} / "
              f"{fmt(s['audio_ratio']['max'], 3)}; reference/native RTF {fmt(s['rtf_ratio']['min'], 3)} / "
              f"{fmt(s['rtf_ratio']['median'], 3)} / {fmt(s['rtf_ratio']['max'], 3)}."]
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-whole', type=Path, required=True)
    parser.add_argument('--native-whole', type=Path, required=True)
    parser.add_argument('--reference-stream', type=Path)
    parser.add_argument('--native-stream', type=Path, required=True)
    parser.add_argument('--reference-streams', action='store_true',
                        help='the reference really streams (first audio before completion); else its TTFA is N/A')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = {name: warm(path) for name, path in (('rw', args.reference_whole), ('nw', args.native_whole),
                                                 ('ns', args.native_stream))}
    runs['rs'] = warm(args.reference_stream) if args.reference_stream else {}
    items = list(runs['nw'])
    per_item = {i: row(runs['rw'].get(i, []), runs['nw'][i], runs['rs'].get(i), runs['ns'].get(i), args.reference_streams)
                for i in items}
    per_bucket = {}
    for bucket in BUCKETS:
        members = [i for i in items if i.startswith(bucket + '-')]
        gather = lambda key: [s for i in members for s in runs[key].get(i, [])]
        per_bucket[bucket] = row(gather('rw'), gather('nw'), gather('rs') or None, gather('ns'), args.reference_streams)
    overall = row([s for v in runs['rw'].values() for s in v], [s for v in runs['nw'].values() for s in v],
                  [s for v in runs['rs'].values() for s in v] or None, [s for v in runs['ns'].values() for s in v],
                  args.reference_streams)
    lengths = {i: {'reference_audio_s_p50': per_item[i]['reference_audio_s_p50'],
                   'native_audio_s_p50': per_item[i]['native_audio_s_p50'],
                   'native_over_reference': ratio(per_item[i]['native_audio_s_p50'], per_item[i]['reference_audio_s_p50'])}
               for i in items}
    view = cutoff_free(runs, items)
    result = {'scope': 'SPEED comparison, warm phase only, failures in every denominator; not an equivalence claim',
              'percentile_convention': 'nearest rank over all attempts, failures ordered after successes; RTF per '
                                       'request = total_s / audio_s; CV and audio medians over successes',
              'per_bucket': per_bucket, 'overall': overall, 'per_item': per_item, 'audio_lengths': lengths,
              'cutoff_free': view}
    with open(args.out, 'x') as out:
        out.write(json.dumps(result, indent=1) + '\n')
    print(markdown('Per length bucket (warm passes pooled)', {**per_bucket, 'all 24 items': overall}))
    print()
    print(cohort_markdown(view))
    print()
    print(markdown('Per item (warm passes)', per_item))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
