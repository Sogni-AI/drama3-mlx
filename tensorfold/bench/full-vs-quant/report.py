#!/usr/bin/env python3
"""Tables, neutral raw records and charts for results/full-vs-quant-01 from a scored run.py run.

Reads RUN_DIR (preflight.json, resume-*-preflight.json and their pin files, ARM/sSALT/{whole,stream,pilot,
health.jsonl,server.sh}, scores.jsonl, pairs.jsonl) and writes OUT_DIR/tables.json, OUT_DIR/tables.md,
OUT_DIR/card-section.md (a model-card section), OUT_DIR/raw/*.jsonl and OUT_DIR/charts/*.{png,svg}. Raw records keep
numbers, item ids and body hashes only: no audio, transcripts of private material, host paths or process identities.

With --grill, also reads a `grill-perf media compare` result (numbers, hashes and reply headers only), copies it
byte-for-byte to OUT_DIR/raw/grill-compare.json, records it in tables.json under `grill` and adds its section to the
card. OUT_DIR/README.md is not written here.

Usage: report.py RUN_DIR OUT_DIR [--grill COMPARE_JSON --grill-collector 'NAME VERSION'
                                  --grill-collector-sha256 HEX --grill-arms a=FG,b=Q,a2=FG]
"""
import argparse
import ast
import hashlib
import json
import math
import random
import re
import textwrap
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'full-vs-quant'  # fixed SVG element ids, so a rerun writes identical SVGs
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GIB = 2 ** 30
RATE, FRAME = 44100, 2048
LABELS = {'F': 'Full FP8', 'Q': 'Quant NVFP4, γ 1.5', 'FG': 'Full FP8, γ 1.5',
          'FB': 'Full FP8, built-in voice', 'QB': 'Quant NVFP4, built-in voice'}
COLORS = {'F': '#4c72b0', 'Q': '#dd8452', 'FG': '#8da0cb', 'FB': '#4c72b0', 'QB': '#dd8452'}
PAIRS = (('Q', 'F'), ('FG', 'F'), ('Q', 'FG'), ('QB', 'FB'))
KV = re.compile(r'(\w+)=([^\s]+)')


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def num(v):
    try:
        return float(v)
    except ValueError:
        return {'true': True, 'false': False}.get(v, v)


def engine_lines(log):
    """Per-request engine records from the server log, in order: the synthesis line, then the speech line."""
    synth, speech = [], []
    for line in Path(log).read_text(errors='replace').splitlines():
        if line.startswith('info: drama: synthesis'):
            synth.append({k: num(v) for k, v in KV.findall(line)})
        elif line.startswith('[tensorfold] speech ') and ' prompt=' in line:
            speech.append({k: num(v) for k, v in KV.findall(line)})
    loaded = re.search(r'loaded in ([0-9.]+)s', Path(log).read_text(errors='replace'))
    return synth, speech, float(loaded.group(1)) if loaded else None


def engine_banner(log):
    """What the engine reports about the loaded model in its start-up line."""
    text = Path(log).read_text(errors='replace')
    line = next(x for x in text.splitlines() if x.startswith('[tensorfold] Drama 3 on CUDA'))
    held = re.search(r'routed experts held (\d+) of (\d+)', line)
    nvfp4 = re.search(r'routed experts NVFP4 in (\d+) of (\d+) layers', line)
    sampling = re.search(r'sampling: (\w+)', text)
    return {'models_gib': float(re.search(r'models ([0-9.]+) GiB', line).group(1)),
            'routed_experts_held': [int(held.group(1)), int(held.group(2))] if held else None,
            'nvfp4_layers': [int(nvfp4.group(1)), int(nvfp4.group(2))] if nvfp4 else None,
            'guidance_on': 'guidance on' in line, 'sampling': sampling.group(1) if sampling else None}


def pins(path):
    """sha256sum pin file as {relative path: digest}, with the ./ and ./model/ prefixes dropped."""
    out = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split(None, 1)
        out[re.sub(r'^(\./)?(model/)?', '', name.strip())] = digest
    return out


def bootstrap_mean_ci(by_item, n=10000, seed=20261022):
    """95 % percentile interval of the mean of per-item values, resampling items."""
    items = list(by_item)
    rng = random.Random(seed)
    means = sorted(mean(v for i in (rng.choice(items) for _ in items) for v in by_item[i]) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for discordant counts b and c."""
    k, n = min(b, c), b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def serve_args(sd):
    """One start's engine arguments, without the checkpoint path, address, served name and voice directory."""
    cmd = json.loads((sd / 'pilot' / 'session.json').read_text())['server_command'][1:]
    wrapper = (sd / 'server.sh').read_text().split('"$@"', 1)[1].split()
    out, skip = [], False
    for a in cmd[:1] + cmd[2:] + wrapper:
        if skip:
            skip = False
        elif a in ('--host', '--port', '--name', '--voices', '--default-voice'):
            skip = True
        else:
            out.append(a)
    return out


def arm_record(run, arm, preflight):
    salts = sorted(run.joinpath(arm).glob('s*'))
    s0 = run / arm / 's0'
    whole = json.loads((s0 / 'whole' / 'summary.json').read_text())
    stream = json.loads((s0 / 'stream' / 'summary.json').read_text())
    loads, engine_loads, peaks, host_drop, raw, engine_raw, starts = [], [], [], [], [], [], {}
    prefill_rate, decode_rate, first_frame = [], [], []
    for sd in salts:
        salt = int(sd.name[1:])
        loads.append(json.loads((sd / 'pilot' / 'ready.json').read_text())['cold_ready_s'])
        synth, speech, loaded = engine_lines(sd / 'pilot' / 'server.log')
        engine_loads.append(loaded)
        health = [h['health'] for h in jl(sd / 'health.jsonl') if 'health' in h]
        for kind in ('whole', 'stream'):
            if (sd / kind / 'server-after.json').exists():
                after = json.loads((sd / kind / 'server-after.json').read_text())
                body = after['health'].get('body')
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except ValueError:
                        body = None
                if isinstance(body, dict) and 'memory' in body:
                    health.append(body)
        peaks.append(max(h['memory']['peak'] for h in health if 'memory' in h))
        mem = jl(sd / 'pilot' / 'memory.jsonl')
        if mem:
            host_drop.append(mem[0]['memavailable_bytes'] - min(m['memavailable_bytes'] for m in mem))
        samples = {k: jl(sd / k / 'samples.jsonl') for k in ('whole', 'stream') if (sd / k / 'samples.jsonl').exists()}
        epochs = [s['started_at_epoch'] for rows in samples.values() for s in rows]
        starts[salt] = [min(epochs), max(epochs)]
        by_format = {'wav': [s for s in speech if s.get('format') == 'wav'],
                     'pcm': [s for s in speech if s.get('format') == 'pcm']}
        for kind, rows in samples.items():
            lines = by_format['wav' if kind == 'whole' else 'pcm']
            # The log carries no request id the client sees: pair by order, and only when every reply's sample count
            # matches the engine line's.
            matched = len(lines) == len(rows) and all(
                not s.get('audio') or abs(line['samples'] / RATE - s['audio']['duration_s']) < 1e-3
                for s, line in zip(rows, lines))
            for i, s in enumerate(rows):
                audio_s = s['audio']['duration_s'] if s.get('audio') else None
                line = lines[i] if matched else {}
                rec = {'arm': arm, 'salt': salt, 'run': kind, 'phase': s['phase'], 'round': s['round'],
                       'item': s['item'], 'outcome': s['outcome'], 'total_s': s['total_s'],
                       'first_frame_s': s.get('first_frame_s'), 'audio_s': audio_s, 'rtf': s.get('rtf'),
                       'body_sha256': s.get('body_sha256'),
                       'prompt_tokens': int(line['prompt']) if 'prompt' in line else None,
                       'finish': line.get('finish'), 'engine_seconds': line.get('seconds')}
                raw.append(rec)
                if kind == 'stream' and s['phase'] == 'warm' and s['outcome'] == 'ok' and s.get('first_frame_s'):
                    frames = audio_s * RATE / FRAME
                    first_frame.append(s['first_frame_s'])
                    decode_rate.append((frames - 1) / (s['total_s'] - s['first_frame_s']))
                    if matched:
                        prefill_rate.append(int(lines[i]['prompt']) / s['first_frame_s'])
        for i, line in enumerate(speech):
            engine_raw.append({'arm': arm, 'salt': salt, 'seq': i, 'line': 'speech', **line})
        for i, line in enumerate(synth):
            engine_raw.append({'arm': arm, 'salt': salt, 'seq': i, 'line': 'synthesis', **line})
    gauges = [h['live'] for sd in salts for h in (x['health'] for x in jl(sd / 'health.jsonl') if 'health' in x)
              if 'live' in h]
    ckpt = preflight['arms'][arm]['checkpoint']
    size = preflight['checkpoints'][ckpt]
    warm, swarm = whole['phases']['warm'], stream['phases']['warm']
    rec = {
        'label': LABELS[arm], 'checkpoint': ckpt, 'flags': preflight['arms'][arm]['flags'],
        'reference_voice': preflight['arms'][arm]['reference_voice'],
        'serve_args': serve_args(s0),
        'request_epoch_span_per_salt': starts,
        'whole_warm_passes_per_salt': {s: len({r['round'] for r in raw if r['salt'] == s and r['run'] == 'whole'
                                               and r['phase'] == 'warm'}) for s in sorted(starts)},
        'streamed_salts': sorted({r['salt'] for r in raw if r['run'] == 'stream'}),
        'cfg_gamma_logged': sorted({line['cfg_gamma'] for line in engine_raw if line['line'] == 'synthesis'}),
        'precision': preflight['arms'][arm]['precision'],
        'engine': engine_banner(s0 / 'pilot' / 'server.log'),
        'weights_and_codec_bytes': size['weights_and_codec_bytes'], 'checkpoint_total_bytes': size['total_bytes'],
        'device_peak_bytes': {'max': max(peaks), 'per_start': peaks},
        'host_memavailable_drop_bytes': {'max': max(host_drop) if host_drop else None, 'per_start': host_drop},
        'cold_ready_s': {'per_start': loads, 'median': median(loads)},
        'engine_loaded_in_s': {'per_start': engine_loads, 'median': median(x for x in engine_loads if x is not None)},
        'whole_warm': {'attempted': warm['attempted'], 'succeeded': warm['succeeded'],
                       'end_to_end_s_p50': warm['end_to_end_s']['ok_only']['p50'],
                       'end_to_end_s_p95': warm['end_to_end_s']['ok_only']['p95'],
                       'rtf_p50': warm['realtime_factor']['ok_only']['p50'],
                       'rtf_p95': warm['realtime_factor']['ok_only']['p95'],
                       'rtf_mean': warm['realtime_factor']['ok_only']['mean'],
                       'audio_s_total': warm['audio_s']['total']},
        'whole_cold_first_request_s': whole['phases']['cold']['end_to_end_s']['ok_only'].get('p50'),
        'whole_warm_rtf_p50_per_salt': {
            s: median(r['rtf'] for r in raw if r['salt'] == s and r['run'] == 'whole' and r['phase'] == 'warm'
                      and r['round'] == 0 and r['outcome'] == 'ok')
            for s in sorted({r['salt'] for r in raw})},
        'stream_warm': {'attempted': swarm['attempted'], 'succeeded': swarm['succeeded'],
                        'first_frame_s_p50': swarm['first_frame_s']['ok_only']['p50'],
                        'first_frame_s_p95': swarm['first_frame_s']['ok_only']['p95'],
                        'decode_frames_per_s_median': median(decode_rate) if decode_rate else None,
                        'prompt_tokens_per_s_to_first_frame_median': median(prefill_rate) if prefill_rate else None,
                        'prompt_tokens_median': median(r['prompt_tokens'] for r in raw if r['prompt_tokens'])
                        if any(r['prompt_tokens'] for r in raw) else None},
        'health_gauges_max': {'prefill_tokens_per_second': max((g.get('prefill_tokens_per_second') or 0) for g in gauges),
                              'decode_tokens_per_second': max((g.get('decode_tokens_per_second') or 0) for g in gauges)},
    }
    return rec, raw, engine_raw


def quality(scores, pairs, arms, finishes):
    out = {}
    for arm in arms:
        rows = [r for r in scores if r['arm'] == arm]
        ok = [r for r in rows if not r.get('missing')]
        by_item = {}
        for r in ok:
            by_item.setdefault(r['item'], []).append(r['wer'])
        sims = [p['cosine'] for p in pairs if p['kind'] == 'arm_vs_full' and p['arm'] == arm]
        per_salt = {}
        for s in sorted({r['salt'] for r in rows}):
            srows = [r for r in rows if r['salt'] == s]
            sok = [r for r in srows if not r.get('missing')]
            per_salt[s] = {'takes': len(srows), 'failed': sum(bool(r['failed']) for r in srows),
                           'wer_mean': mean(r['wer'] for r in sok) if sok else None}
        finish = [finishes.get((arm, r['salt'], r['item'])) for r in rows]
        out[arm] = {'takes': len(rows), 'missing': len(rows) - len(ok),
                    'failed': sum(bool(r['failed']) for r in rows), 'capped': sum(bool(r.get('capped')) for r in ok),
                    'collapsed': sum(bool(r.get('collapsed')) for r in ok),
                    'engine_finish': {f: finish.count(f) for f in sorted(set(finish), key=str)},
                    'wer_mean': mean(r['wer'] for r in ok), 'wer_mean_ci95': bootstrap_mean_ci(by_item),
                    'wer_pooled': sum(r['wer'] * r['ref_words'] for r in ok) / sum(r['ref_words'] for r in ok),
                    'wer_median': median(r['wer'] for r in ok),
                    'per_salt': per_salt,
                    'quiet_run_s_max': max(r['quiet_run_s'] for r in ok),
                    'audio_s_mean': mean(r['audio_s'] for r in ok),
                    'speaker_cosine_vs_full_mean': mean(sims) if sims else None}
    for base in sorted({p['arm'] for p in pairs if p['kind'] == 'full_across_salts'}):
        sims = [p['cosine'] for p in pairs if p['kind'] == 'full_across_salts' and p['arm'] == base]
        out[base]['speaker_cosine_same_model_across_salts_mean'] = mean(sims)
    paired = {}
    index = {(r['arm'], r['salt'], r['item']): r for r in scores}
    for a, b in PAIRS:
        keys = [(s, i) for (x, s, i) in index if x == a and (b, s, i) in index]
        if not keys:
            continue
        fa = [bool(index[(a, s, i)]['failed']) for s, i in keys]
        fb = [bool(index[(b, s, i)]['failed']) for s, i in keys]
        only_a = sum(x and not y for x, y in zip(fa, fb))
        only_b = sum(y and not x for x, y in zip(fa, fb))
        diff = {}
        for s, i in keys:
            ra, rb = index[(a, s, i)], index[(b, s, i)]
            if not ra.get('missing') and not rb.get('missing'):
                diff.setdefault(i, []).append(ra['wer'] - rb['wer'])
        paired[f'{a}_vs_{b}'] = {'pairs': len(keys), 'failed_only_' + a: only_a, 'failed_only_' + b: only_b,
                                 'mcnemar_exact_p': mcnemar_exact(only_a, only_b),
                                 'wer_diff_mean': mean(v for vs in diff.values() for v in vs),
                                 'wer_diff_ci95_items': bootstrap_mean_ci(diff)}
    return out, paired


def charts(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ('F', 'FG', 'Q') if a in tables['arms']]
    built = [a for a in ('FB', 'QB') if a in tables['arms']]
    A = tables['arms']
    Qy = tables['quality']

    def bars(ax, names, values, title, unit, fmt='{:.2f}', tops=None):
        xs = range(len(names))
        b = ax.bar(xs, values, color=[COLORS[n] for n in names], hatch=None)
        for n, rect in zip(names, b):
            if n in ('FG',) or n.endswith('B'):
                rect.set_hatch('//' if n == 'FG' else '..')
        ax.set_xticks(list(xs), [textwrap.fill(LABELS[n].replace('γ ', 'γ\u00a0'), 13) for n in names], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        for x, v, top in zip(xs, values, tops or values):
            ax.annotate(fmt.format(v), (x, top), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom',
                        fontsize=8)
        ax.margins(y=0.15)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f'{name}.png', dpi=150)
        fig.savefig(out / f'{name}.svg', metadata={'Date': None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    fq = [a for a in ('F', 'Q') if a in A]
    bars(axes[0], fq, [A[a]['weights_and_codec_bytes'] / GIB for a in fq], 'Weights + codec on disk', 'GiB', '{:.1f}')
    bars(axes[1], arms, [A[a]['device_peak_bytes']['max'] / GIB for a in arms], 'Peak device memory (/health)', 'GiB',
         '{:.1f}')
    fig.suptitle('Drama 3 on one GB10: footprint', fontsize=11)
    save(fig, 'footprint')

    names = arms + built
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    bars(axes[0], names, [A[a]['whole_warm']['rtf_p50'] for a in names], 'Real-time factor, warm p50, salt 0\n'
         '(lower is faster)', 'RTF')
    bars(axes[1], names, [A[a]['stream_warm']['first_frame_s_p50'] for a in names],
         'First audio when streaming, p50, salt 0', 's')
    bars(axes[2], names, [A[a]['stream_warm']['decode_frames_per_s_median'] for a in names],
         'Decode, codec frames/s (median)', 'frames/s', '{:.1f}')
    bars(axes[3], names, [A[a]['cold_ready_s']['median'] for a in names], 'Process start to ready (cold disk)', 's',
         '{:.1f}')
    fig.suptitle('Speed: one engine binary, 24-item corpus, concurrency 1', fontsize=11)
    save(fig, 'speed')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    wer = [Qy[a]['wer_mean'] * 100 for a in names]
    tops = [max([Qy[a]['wer_mean_ci95'][1]] + [v['wer_mean'] for v in Qy[a]['per_salt'].values()]) * 100
            for a in names]
    bars(axes[0], names, wer, 'Word error rate, Whisper large-v3 (mean,\n95 % CI over texts, dots: each salt)', '%',
         '{:.1f}', tops)
    for x, a in enumerate(names):
        lo, hi = Qy[a]['wer_mean_ci95']
        axes[0].errorbar(x, Qy[a]['wer_mean'] * 100, yerr=[[Qy[a]['wer_mean'] * 100 - lo * 100],
                                                           [hi * 100 - Qy[a]['wer_mean'] * 100]], fmt='none',
                         ecolor='k', capsize=3, lw=1)
        salt_wer = [v['wer_mean'] * 100 for v in Qy[a]['per_salt'].values()]
        axes[0].scatter([x + 0.22] * len(salt_wer), salt_wer, s=10, color='k', zorder=3)
    bars(axes[1], names, [Qy[a]['failed'] / Qy[a]['takes'] * 100 for a in names], 'Failed takes (cap, WER > 0.2 '
         'or ≥ 3 s stall)', '% of takes', '{:.1f}')
    sim = [a for a in names if Qy[a].get('speaker_cosine_vs_full_mean') is not None]
    bars(axes[2], sim, [Qy[a]['speaker_cosine_vs_full_mean'] for a in sim], 'Speaker similarity to the full\nmodel\'s '
         'take (ECAPA cosine)', 'cosine', '{:.3f}')
    for base, ls in (('F', '--'), ('FB', ':')):
        v = Qy.get(base, {}).get('speaker_cosine_same_model_across_salts_mean')
        if v is not None:
            axes[2].axhline(v, color='k', ls=ls, lw=1, label=f'{LABELS[base]} vs itself, other salt')
    axes[2].legend(fontsize=7, loc='upper right')
    axes[2].set_ylim(0, 1.3)
    fig.suptitle('Quality: 24 items × 4 salts per arm', fontsize=11)
    save(fig, 'quality')


def markdown(t):
    A, Qy, P = t['arms'], t['quality'], t['paired']
    names = [a for a in ('F', 'Q', 'FG', 'FB', 'QB') if a in A]
    head = '| Measure | ' + ' | '.join(LABELS[a] for a in names) + ' |\n|---|' + '---:|' * len(names) + '\n'

    def row(name, f):
        return f'| {name} | ' + ' | '.join(f(a) for a in names) + ' |\n'
    g = lambda b: f'{b / GIB:.2f}'  # noqa: E731
    md = head
    md += row('Weights + codec on disk (GiB)', lambda a: g(A[a]['weights_and_codec_bytes']))
    md += row('Peak device memory, /health (GiB)', lambda a: g(A[a]['device_peak_bytes']['max']))
    md += row('Host MemAvailable drop, max (GiB)', lambda a: g(A[a]['host_memavailable_drop_bytes']['max']))
    md += row('Process start to ready, arithmetic median of 4 cold starts (s)',
              lambda a: f'{A[a]["cold_ready_s"]["median"]:.1f}')
    md += row('Engine load, arithmetic median (s)', lambda a: f'{A[a]["engine_loaded_in_s"]["median"]:.1f}')
    md += row('Real-time factor, warm p50 nearest-rank, salt 0 (all warm passes)',
              lambda a: f'{A[a]["whole_warm"]["rtf_p50"]:.3f}')
    md += row('Real-time factor, warm p95 nearest-rank, salt 0 (all warm passes)',
              lambda a: f'{A[a]["whole_warm"]["rtf_p95"]:.3f}')
    md += row('Real-time factor, arithmetic median of each salt\'s first warm pass, lowest–highest', lambda a: (
        f'{min(A[a]["whole_warm_rtf_p50_per_salt"].values()):.3f}–'
        f'{max(A[a]["whole_warm_rtf_p50_per_salt"].values()):.3f}'))
    md += row('Whole reply, warm p50 nearest-rank, salt 0 (s)',
              lambda a: f'{A[a]["whole_warm"]["end_to_end_s_p50"]:.2f}')
    md += row('First audio when streaming, p50 nearest-rank, salt 0 (s)',
              lambda a: f'{A[a]["stream_warm"]["first_frame_s_p50"]:.3f}')
    md += row('Decode, codec frames/s, streamed, salt 0 (arithmetic median)',
              lambda a: f'{A[a]["stream_warm"]["decode_frames_per_s_median"]:.1f}')
    md += row('Prompt tokens per second to first audio (arithmetic median)', lambda a: (
        f'{A[a]["stream_warm"]["prompt_tokens_per_s_to_first_frame_median"]:.0f}'
        if A[a]['stream_warm']['prompt_tokens_per_s_to_first_frame_median'] else '—'))
    md += row('Prompt tokens per request (arithmetic median)', lambda a: (
        f'{A[a]["stream_warm"]["prompt_tokens_median"]:.0f}' if A[a]['stream_warm']['prompt_tokens_median'] else '—'))
    md += row('Warm requests succeeded', lambda a: f'{A[a]["whole_warm"]["succeeded"]}/{A[a]["whole_warm"]["attempted"]}'
              f' + {A[a]["stream_warm"]["succeeded"]}/{A[a]["stream_warm"]["attempted"]} streamed')
    md += row('Word error rate, mean [95 % CI]', lambda a: f'{Qy[a]["wer_mean"] * 100:.2f} % '
              f'[{Qy[a]["wer_mean_ci95"][0] * 100:.2f}, {Qy[a]["wer_mean_ci95"][1] * 100:.2f}]')
    md += row('Word error rate, pooled', lambda a: f'{Qy[a]["wer_pooled"] * 100:.2f} %')
    md += row('Word error rate, mean of each salt, lowest–highest', lambda a: (
        f'{min(v["wer_mean"] for v in Qy[a]["per_salt"].values()) * 100:.2f}–'
        f'{max(v["wer_mean"] for v in Qy[a]["per_salt"].values()) * 100:.2f} %'))
    md += row('Failed takes (cap, collapsed)', lambda a: f'{Qy[a]["failed"]}/{Qy[a]["takes"]} (cap {Qy[a]["capped"]}, '
              f'collapsed {Qy[a]["collapsed"]})')
    md += row('Failed takes per salt (0 / 1 / 2 / 3)', lambda a: ' / '.join(
        str(Qy[a]['per_salt'][s]['failed']) for s in sorted(Qy[a]['per_salt'])))
    md += row('Engine finish reason, scored takes', lambda a: ', '.join(
        f'{k if k is not None else "unmatched"} {v}' for k, v in Qy[a]['engine_finish'].items()))
    md += row('Speaker cosine to the full model\'s take', lambda a: (
        f'{Qy[a]["speaker_cosine_vs_full_mean"]:.3f}' if Qy[a].get('speaker_cosine_vs_full_mean') is not None else
        f'{Qy[a]["speaker_cosine_same_model_across_salts_mean"]:.3f} (itself, other salt)'))
    md += ('\nTwo median conventions. Rows marked nearest-rank are the benchmark client\'s own summaries '
           '(`bench/drama_bench.py`, `percentiles`): the sorted value at rank ⌈p·n/100⌉, never interpolated, so a p50 '
           'over an even count is the lower of the two middle values. Rows marked arithmetic median are computed '
           'by `bench/full-vs-quant/report.py` with Python\'s `statistics.median`, which averages the two middle '
           'values of an even count; the per-salt real-time-factor range is one of these (its `tables.json` key keeps '
           'the name `whole_warm_rtf_p50_per_salt`).\n')
    md += '\n| Paired comparison | Pairs | Failed only in first | Failed only in second | Exact McNemar p | ' \
          'WER difference, mean [95 % CI over items] |\n|---|---:|---:|---:|---:|---|\n'
    for k, v in P.items():
        a, b = k.split('_vs_')
        md += (f'| {LABELS[a]} vs {LABELS[b]} | {v["pairs"]} | {v["failed_only_" + a]} | {v["failed_only_" + b]} | '
               f'{v["mcnemar_exact_p"]:.3f} | {v["wer_diff_mean"] * 100:+.2f} pp [{v["wer_diff_ci95_items"][0] * 100:+.2f}, '
               f'{v["wer_diff_ci95_items"][1] * 100:+.2f}] |\n')
    return md


def pin_file(run, verify):
    """The pin file a preflight verified with, found in the run or in recipe/checksums, only if its digest matches."""
    name = Path(verify['pins']).name
    for path in (run / name, REPO / 'recipe' / 'checksums' / name):
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == verify['pins_sha256']:
            return path
    return None


def under(paths):
    """{top two path components or file name: count}, and whether every path is hidden (a component starts with .)."""
    counts = {}
    for p in paths:
        key = '/'.join(p.split('/')[:2]) + '/' if '/' in p else p
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items())), all(any(s.startswith('.') for s in p.split('/')) for p in paths)


def resume_record(run, first, again):
    """What a resume repeated from the first preflight, and how its checkpoint pins changed."""
    rec = {'started_at': again['started_at'], 'harness_commit': again['harness_commit'],
           'harness_dirty': again['harness_dirty'],
           'same_as_first': {k: again[k] == first[k] for k in ('binary_sha256', 'voice_sha256', 'corpus_sha256',
                                                              'arms', 'salts')},
           'checkpoint_pins_sha256': {k: v['verify']['pins_sha256'] for k, v in again['checkpoints'].items()},
           'checkpoint_files_verified': {k: v['verify']['files'] for k, v in again['checkpoints'].items()},
           'checkpoint_verify_returncode': {k: v['verify']['returncode'] for k, v in again['checkpoints'].items()},
           'pin_changes': {}}
    for name, v in again['checkpoints'].items():
        a, b = first['checkpoints'][name]['verify'], v['verify']
        if a['pins_sha256'] == b['pins_sha256']:
            continue
        fa, fb = pin_file(run, a), pin_file(run, b)
        if not (fa and fb):
            rec['pin_changes'][name] = {'pin_files_found': False}
            continue
        pa, pb = pins(fa), pins(fb)
        dropped = sorted(set(pa) - set(pb))
        dropped_under, dropped_hidden = under(dropped)
        rec['pin_changes'][name] = {
            'pin_files_found': True, 'entries_first': len(pa), 'entries_resume': len(pb),
            'entries_dropped': len(dropped), 'entries_added': len(set(pb) - set(pa)),
            'dropped_under': dropped_under, 'dropped_all_hidden': dropped_hidden,
            'kept_entries_same_sha256': all(pa.get(k) == h for k, h in pb.items())}
        client = fb.with_name(fb.name.replace('.sha256', '-visible.sha256'))
        if client.exists():
            pc = pins(client)
            rec['pin_changes'][name]['client_pins'] = {
                'entries': len(pc), 'hidden_entries': sum(any(s.startswith('.') for s in p.split('/')) for p in pc),
                'same_sha256_as_resume_pins': all(pb.get(k) == h for k, h in pc.items()),
                'resume_entries_left_out': under(sorted(set(pb) - set(pc)))[0]}
    return rec


def set_aside(run):
    """Arm/salt directories a resume moved aside, with what the benchmark client refused (counts only)."""
    out = []
    for p in sorted(run.glob('*/resume-*-aside-*')):
        rec = {'path': str(p.relative_to(run)), 'requests_sent': (p / 'whole' / 'samples.jsonl').exists()}
        text = (p / 'whole.out').read_text(errors='replace') if (p / 'whole.out').exists() else ''
        refused = re.search(r'refused: checkpoint differs from its pins: mismatched (\[.*?\]), missing (\[.*?\])', text)
        if refused:
            missing = ast.literal_eval(refused.group(2))
            missing_under, missing_hidden = under(missing)
            rec['refusal'] = {'mismatched': len(ast.literal_eval(refused.group(1))), 'missing': len(missing),
                              'missing_under': missing_under, 'missing_all_hidden': missing_hidden}
        out.append(rec)
    return out


def fraction(x):
    return x['numerator'] / x['denominator']


def grill_record(compare, collector, collector_sha256, slot_arms):
    """The numbers of a `grill-perf media compare` result, with the collector and run-01 arm each slot served."""
    c = json.loads(compare)
    slots = {}
    for a in c['acquisitions']:
        replies = [r['reply'] for r in a['runs']]
        headers = [dict(r['headers']) for r in replies]
        dates = sorted(parsedate_to_datetime(h['date']) for h in headers)
        slots[a['slot']['id']] = {
            'status': a['status'], 'cache_mutated_during_measurement': a['cache_mutated_during_measurement'],
            'replies': len(replies), 'seeds': sorted({r['seed'] for r in a['runs']}),
            'http_status': sorted({r['http_status'] for r in replies}),
            'failures': sum(r['failure'] is not None for r in replies),
            'content_type': sorted({h.get('content-type', '') for h in headers}),
            'x_cfg_gamma': sorted({h.get('x-cfg-gamma', '') for h in headers}),
            'reply_date_first': dates[0].isoformat(), 'reply_date_last': dates[-1].isoformat(),
            'body_bytes_max': max(r['body']['bytes'] for r in replies),
            'bodies_truncated': sum(r['body']['truncated'] for r in replies),
            'cells': {x['cell']: {'warmup_runs': x['warmup_runs'], 'measured_runs': x['measured_runs'],
                                  'eligible': x['eligible'], 'ineligible_by_reason': x['ineligible_by_reason'],
                                  'finish_reason_unknown': x['finish_reason_unknown'],
                                  'deterministic_per_seed': all(d['deterministic'] for d in x['determinism_by_seed']),
                                  'first_audio_us_median': x['first_audio_us']['median'],
                                  'end_to_end_us_median': x['end_to_end_us']['median']} for x in a['cells']}}
    gates, rtf = [], {}
    for g in c['gates']:
        cell = g['gate']['cell']
        rs, cs = g['reference_summary'], g['candidate_summary']
        rtf[cell] = {'reference': rs['realtime_factor'], 'candidate': cs['realtime_factor']}
        gates.append({'cell': cell, 'metric': g['gate']['metric'], 'outcome': g['outcome'], 'reason': g['reason'],
                      'max_adverse_bps': g['gate']['max_adverse_bps'],
                      'max_reference_spread_bps': g['gate']['max_reference_spread_bps'],
                      'reference_range': [fraction(x) for x in g['reference']],
                      'candidate_range': [fraction(x) for x in g['candidate']],
                      'adverse_bounds': g['adverse_bounds'],
                      'reference_eligible': rs['eligible'], 'candidate_eligible': cs['eligible'],
                      'audio_bytes_equal_across_arms': [e['equal'] for e in g['cross_arm_hash_equality']]})
    return {'compare_sha256': hashlib.sha256(compare).hexdigest(), 'copy': 'raw/grill-compare.json',
            'collector': collector, 'collector_binary_sha256': collector_sha256, 'slot_arms': slot_arms,
            **{k: c[k] for k in ('kind', 'version', 'claim', 'support', 'axis', 'workload_sha256', 'outcome',
                                 'reasons', 'reference_identity')},
            'slots': slots, 'gates': gates, 'realtime_factor': rtf}


# Media compare exit codes as the collector documents them (grill-perf MEDIA.md, v0.11.0).
GRILL_EXIT = {'PASS': 0, 'ERROR': 1, 'INVALID': 1, 'INCONCLUSIVE': 2, 'REGRESSION': 3}
GRILL_METRIC = {'first_audio_us': 'First audio', 'end_to_end_us': 'End-to-end', 'duration_ratio': 'Duration ratio'}


def grill_card(t):
    """Card section for The Grill's full-against-quant comparison: what it says, and how it sits beside run-01."""
    G, A = t['grill'], t['arms']
    S, arms = G['slots'], G['slot_arms']
    ref, cand = arms['a'], arms['b']
    if arms.get('a2') != ref:
        raise SystemExit('--grill-arms: a and a2 must name the same run-01 arm')
    cells = list(S['a']['cells'])
    gate = {(g['cell'], g['metric']): g for g in G['gates']}
    sec = lambda us: us / 1e6  # noqa: E731
    pct = lambda x: f'{x * 100:+.1f} %'  # noqa: E731
    cfg = ', '.join(sorted({v for s in S.values() for v in s['x_cfg_gamma']}))
    ctype = ', '.join(sorted({v for s in S.values() for v in s['content_type']}))
    seeds = ', '.join(str(x) for x in sorted({v for s in S.values() for v in s['seeds']}))
    first = S['a']['cells'][cells[0]]
    md = '### Speed under The Grill (separate from run-01)\n\n'
    md += (f'Measured with The Grill: `{G["collector"]}` (binary sha256 `{G["collector_binary_sha256"][:12]}…`, as '
           f'given to `report.py`), three `media capture` slots and one `media compare` (`{G["kind"]}`, version '
           f'{G["version"]}). Slots a and a2 served the configuration of run-01 arm {ref} ({LABELS[ref]}, '
           f'checkpoint {A[ref]["checkpoint"]}) and slot b that of arm {cand} ({LABELS[cand]}, checkpoint '
           f'{A[cand]["checkpoint"]}); the comparison pools a and a2 as the reference (A + A2) against b (B). Every '
           f'reply in the comparison carries `x-cfg-gamma: {cfg}` and `content-type: {ctype}` (streamed). Cells: '
           f'{", ".join(cells)} from the '
           f'bench corpus, seed {seeds}, {first["warmup_runs"]} warm-up and {first["measured_runs"]} measured requests '
           f'per cell and slot. Capture periods (reply Date headers, UTC): '
           + '; '.join(f'{k} {s["reply_date_first"][:10]} {s["reply_date_first"][11:19]}–{s["reply_date_last"][11:19]}'
                       for k, s in S.items())
           + f'. The comparison is `{G["copy"]}` (sha256 `{G["compare_sha256"][:12]}…`), byte for byte; every Grill '
           f'number in this section comes from it, and every run-01 number from the tables above. The workload file it '
           f'names has sha256 `{G["workload_sha256"][:12]}…`. This makes no claim about concurrent load.\n\n')
    md += (f'- **Claim scope**, as the comparison states it: `{G["claim"]}`, on the `{G["axis"]}` axis, support level '
           f'`{G["support"]}`: The Grill is early experimental software. Reference identity '
           f'`{G["reference_identity"]["status"]}`: "{G["reference_identity"]["scope"]}"\n')
    complete = all(s['status'] == 'complete' for s in S.values())
    clean = all(s['failures'] == 0 and s['http_status'] == [200] for s in S.values())
    eligible = all(x['eligible'] == x['measured_runs'] and not x['ineligible_by_reason']
                   for s in S.values() for x in s['cells'].values())
    md += (f'- **Outcome: {G["outcome"]}** (reason `{", ".join(G["reasons"])}`; exit '
           f'{GRILL_EXIT[G["outcome"]]} in the collector\'s mapping: 0 PASS, 1 ERROR or INVALID, 2 INCONCLUSIVE, '
           f'3 REGRESSION).')
    if G['axis'] == 'changed-model' and G['reasons'] == ['changed_model_axis']:
        md += (' The slots serve different models, which is the point of this comparison; on that axis the collector '
               'reports each gate but never an overall pass, so INCONCLUSIVE is the result by design.')
    status = ', '.join(f'{k} {s["status"]}' for k, s in S.items())
    replies = ', '.join(f'{k} {s["replies"]}' for k, s in S.items())
    md += (f' Captures: acquisition status {status}; replies {replies}, HTTP status '
           f'{", ".join(map(str, sorted({c for s in S.values() for c in s["http_status"]})))}, '
           f'{sum(s["failures"] for s in S.values())} failed; measured requests eligible in every cell and slot: '
           f'{"all" if eligible else "NOT all"}.'
           + (' It is not a failed capture.' if complete and clean and eligible else '') + '\n')
    md += '\n| Cell | Gate | Adverse bound | A + A2, min–max | B, min–max | Adverse-bound interval | Outcome |\n'
    md += '|---|---|---:|---:|---:|---:|---|\n'
    for g in G['gates']:
        if g['metric'] == 'duration_ratio':
            v = lambda x: f'{x:.3f}'  # noqa: E731
            iv = f'{g["adverse_bounds"][0]:.3f}' if g['adverse_bounds'][0] == g['adverse_bounds'][1] else \
                f'{g["adverse_bounds"][0]:.3f} to {g["adverse_bounds"][1]:.3f}'
        else:
            v = lambda x: f'{sec(x):.3f} s'  # noqa: E731
            iv = f'{pct(g["adverse_bounds"][0])} to {pct(g["adverse_bounds"][1])}'
        bound = (f'{g["max_adverse_bps"] / 100:.2f} %' if g['metric'] != 'duration_ratio'
                 else f'{g["max_adverse_bps"] / 10000:.3f}')
        md += (f'| {g["cell"]} | {GRILL_METRIC.get(g["metric"], g["metric"])} | {bound} | '
               f'{v(g["reference_range"][0])}–{v(g["reference_range"][1])} (n {g["reference_eligible"]}) | '
               f'{v(g["candidate_range"][0])}–{v(g["candidate_range"][1])} (n {g["candidate_eligible"]}) | {iv} | '
               f'{g["outcome"]} |\n')
    md += ('\nTiming gates: the interval runs from B\'s minimum against A + A2\'s maximum to B\'s maximum against '
           'their minimum, as a relative change; negative means B was lower, positive would count against the bound. '
           'Duration ratio is B\'s audio length over A + A2\'s for the same text and seed; a length change in either '
           'direction is adverse, against a bound of '
           f'{gate[(cells[0], "duration_ratio")]["max_adverse_bps"] / 10000:.3f}. Every gate\'s reference-spread limit '
           f'(`max_reference_spread_bps`) is '
           + ', '.join(sorted({f'{g["max_reference_spread_bps"] / 100:.0f} %' for g in G['gates']}))
           + '; the a and a2 medians are in the next table.\n\n')
    md += ('| Cell | First audio, median a / a2 / b (s) | End-to-end, median a / a2 / b (s) | Real-time factor, '
           'A + A2 / B (not gated) | B audio length / A + A2 |\n|---|---:|---:|---:|---:|\n')
    for cell in cells:
        fa = ' / '.join(f'{sec(S[k]["cells"][cell]["first_audio_us_median"]):.3f}' for k in ('a', 'a2', 'b'))
        e2e = ' / '.join(f'{sec(S[k]["cells"][cell]["end_to_end_us_median"]):.3f}' for k in ('a', 'a2', 'b'))
        r = G['realtime_factor'][cell]
        md += (f'| {cell} | {fa} | {e2e} | {r["reference"]:.3f} / {r["candidate"]:.3f} | '
               f'{gate[(cell, "duration_ratio")]["candidate_range"][0]:.3f} |\n')
    passed = lambda m: [c for c in cells if gate[(c, m)]['outcome'] == 'PASS']  # noqa: E731
    failed_dur = [c for c in cells if gate[(c, 'duration_ratio')]['outcome'] != 'PASS']
    ratios = [f'{gate[(c, "duration_ratio")]["candidate_range"][0]:.3f}' for c in failed_dur]
    adverse = [f'{gate[(c, "duration_ratio")]["adverse_bounds"][1]:.3f}' for c in failed_dur]
    rtf_change = {c: G['realtime_factor'][c]['candidate'] / G['realtime_factor'][c]['reference'] - 1 for c in cells}
    md += (f'\nFirst audio passes in {", ".join(passed("first_audio_us")) or "no cell"} and end-to-end time in '
           f'{", ".join(passed("end_to_end_us")) or "no cell"}. Duration ratio '
           + (f'is {"/".join(sorted({gate[(c, "duration_ratio")]["outcome"] for c in failed_dur}))} in '
              f'{" and ".join(failed_dur)}: B\'s audio is {" and ".join(ratios)} of A + A2\'s length there, an '
              f'adverse deviation of {" and ".join(adverse)} against a bound of '
              f'{gate[(failed_dur[0], "duration_ratio")]["max_adverse_bps"] / 10000:.3f}'
              if failed_dur else 'passes in every cell')
           + (f'; it passes in {", ".join(passed("duration_ratio"))} ('
              + ', '.join(f'{gate[(c, "duration_ratio")]["candidate_range"][0]:.3f}' for c in passed('duration_ratio'))
              + ')' if failed_dur and passed('duration_ratio') else '')
           + '. Shorter audio is not a speed-up: end-to-end time grows with the audio produced, so where B\'s audio is '
           'shorter part of its lower end-to-end time is less audio, not faster synthesis. The real-time factor '
           '(end-to-end time over audio time, reported by the collector, not gated) is the length-normalised figure: '
           f'B against A + A2 {", ".join(f"{pct(rtf_change[c])} in {c}" for c in cells)}. ')
    equal = [e for g in G['gates'] for e in g['audio_bytes_equal_across_arms']]
    determ = all(x['deterministic_per_seed'] for s in S.values() for x in s['cells'].values())
    unknown = sum(x['finish_reason_unknown'] for s in S.values() for x in s['cells'].values())
    body_max = max(s['body_bytes_max'] for s in S.values())
    md += (f'B\'s audio bytes are {"never" if not any(equal) else "sometimes"} equal to A\'s for the same seed, as '
           f'expected across a model change; within each slot every cell was '
           f'{"deterministic per seed" if determ else "NOT deterministic per seed"}. The finish reason is unknown on '
           f'{unknown} measured replies (the engine sends it as an HTTP trailer and the collector reads headers), so '
           f'the collector cannot flag a clipped reply; the largest body is {body_max:,} bytes, below the '
           f'{1024 * FRAME * 2:,} bytes of a 1024-frame reply.\n\n')
    md += ('- **What it is not.** It measures its own declared cells and bounds only; it says nothing about output '
           f'quality (for these two configurations see the {cand} and {ref} columns above), and none of its numbers is '
           f'pooled into run-01\'s tables or charts.\n')
    rows = (('real-time factor p50', lambda a: A[a]['whole_warm']['rtf_p50'], '{:.3f}'),
            ('first audio p50', lambda a: A[a]['stream_warm']['first_frame_s_p50'], '{:.3f} s'),
            ('whole reply p50', lambda a: A[a]['whole_warm']['end_to_end_s_p50'], '{:.2f} s'))
    run01 = [(n, f(cand), f(ref), fmt) for n, f, fmt in rows]
    grill_lower = all(g['adverse_bounds'][1] < 0 for g in G['gates'] if g['metric'] != 'duration_ratio') and \
        all(v < 0 for v in rtf_change.values())
    run01_lower = all(c < r for _, c, r, _ in run01)
    md += ('- **Beside run-01.** For the same two configurations run-01\'s salt-0 rows '
           + (f'point the same way, lower for {cand}' if grill_lower and run01_lower else
              'do NOT all point the same way as The Grill')
           + f': {cand} against {ref}, '
           + ', '.join(f'{n} {fmt.format(c)} against {fmt.format(r)} ({pct(c / r - 1)})' for n, c, r, fmt in run01)
           + f'; {cand}\'s warm whole replies at salt 0 hold {A[cand]["whole_warm"]["audio_s_total"]:.0f} s of audio '
           f'against {A[ref]["whole_warm"]["audio_s_total"]:.0f} s '
           f'({pct(A[cand]["whole_warm"]["audio_s_total"] / A[ref]["whole_warm"]["audio_s_total"] - 1)}). The '
           f'workloads differ, so the sizes are not interchangeable: run-01 is '
           f'{t["quality"][ref]["per_salt"][0]["takes"]} texts at seed salt 0, whole WAV replies for the real-time '
           f'factor and whole-reply rows and streamed replies for first audio, with nearest-rank p50s; The Grill is '
           f'{len(cells)} texts at seed {seeds}, streamed PCM replies only, a and a2 pooled, with the collector\'s '
           f'medians and min–max ranges.\n\n')
    return md


def card(t):
    """Model-card section: what the recipe is, the tables, the charts, how quality was scored and the run's history."""
    A, Qy, E = t['arms'], t['quality'], t['environment']
    F, Q, b = A['F'], A['Q'], E['engine_build']
    g = lambda n: f'{n / GIB:.2f} GiB'  # noqa: E731
    voice = lambda a: 'reference voice' if A[a]['reference_voice'] else 'built-in voice'  # noqa: E731
    salts = E['salts']
    items = Qy['F']['per_salt'][salts[0]]['takes']
    passes = F['whole_warm_passes_per_salt']
    md = '## Full model against this quant\n\n'
    md += (f'One NVIDIA GB10, one engine binary for every arm, the {items} English texts of the bench corpus '
           f'(`bench/corpus/drama-bench-en-v1.jsonl`, sha256 `{E["corpus_sha256"][:12]}…`) one request at a time, '
           f'and {len(salts)} seed salts ({", ".join(map(str, salts))}) per arm. Every number below comes from the '
           f'files of run `run-01` through `bench/full-vs-quant/report.py`'
           + ('; The Grill\'s section takes its own from `raw/grill-compare.json`' if 'grill' in t else '')
           + '; the per-request records are in `raw/`.\n\n')
    md += '| Arm | Checkpoint | Voice | Engine arguments |\n|---|---|---|---|\n'
    for a in A:
        md += f'| {LABELS[a]} | {A[a]["checkpoint"]} | {voice(a)} | `{" ".join(A[a]["serve_args"])}` |\n'
    held, layers = Q['engine']['routed_experts_held'], Q['engine']['nvfp4_layers']
    md += '\n### The recipe\n\n'
    md += (f'- **Quant.** {Q["precision"].split(";")[0]}. At start-up the engine reports '
           f'{Q["engine"]["models_gib"]:.2f} '
           f'GiB of model weights against {F["engine"]["models_gib"]:.2f} GiB for the full FP8 checkpoint, '
           f'{held[0]:,} of {held[1]:,} routed experts held and NVFP4 routed experts in {layers[0]} of {layers[1]} '
           f'layers. Weights and codec on disk: {g(Q["weights_and_codec_bytes"])} against '
           f'{g(F["weights_and_codec_bytes"])}.\n')
    gammas = {a: A[a]['cfg_gamma_logged'] for a in A}
    inactive = [a for a in A if A[a]['flags'] and gammas[a] == [1.0]]
    md += (f'- **Engine.** TensorFold native (Zig/CUDA), build `{b["describe"]}` from branch `{b["branch"]}`, '
           f'clean tree: '
           f'{"yes" if b["clean_tree"] else "no"}; binary sha256 `{E["engine_binary_sha256"][:12]}…`. Sampling as the '
           f'engine reports it: {", ".join(sorted({A[a]["engine"]["sampling"] for a in A}))}. The quant is served with '
           f'guidance γ 1.5 (`--cfg-gamma 1.5 --cfg-batched`): a second, unconditioned backbone sequence run in one '
           f'batch with the first. The full-model arm with the same guidance (FG) separates the effect of guidance '
           f'from the effect of the quant.'
           + (f' With the built-in voice the engine logged γ 1.00 on every reply of {", ".join(inactive)}, so guidance '
              f'was inactive there.' if inactive else '')
           + ' The engine arguments in the table above are complete; none sets a redraw or retry.\n')
    codec = E['codec_pth_sha256']
    if None not in codec.values() and len(set(codec.values())) == 1:
        md += (f'- **Codec.** The quant ships Drama 3\'s codec weights unchanged: `codec.pth` has the same sha256 '
               f'(`{codec["quant"][:12]}…`) in both checkpoints\' pins. ')
    else:
        md += f'- **Codec.** `codec.pth` sha256 per checkpoint: {codec}. '
    md += ('The engine binary decodes it with codec scratch kept per chunk, a host-side change (the build record is '
           'in `tables.json` under `environment.engine_build.change`).\n\n')
    md += '### Results\n\n' + markdown(t) + '\n'
    md += (f'Real-time factor is client end-to-end time over audio time (lower is faster). Whole warm passes per salt: '
           f'{" / ".join(str(passes[s]) for s in salts)}; streamed warm passes ran only in salt(s) '
           f'{", ".join(map(str, F["streamed_salts"]))} ({F["stream_warm"]["attempted"] // items} per text). '
           f'Rows marked salt 0 use only that salt. Quality scores only the first warm pass of '
           f'each salt\'s whole replies: {items} takes per salt. A take fails if it reaches the 1024-frame cap or '
           f'collapses (word error rate above 0.2, or a silence of 3 s or more inside the speech). The 95 % intervals '
           f'resample the {items} texts 10,000 times. Speaker cosine compares ECAPA embeddings with the full model\'s '
           f'take of the same text and salt (F for the reference voice, FB for the built-in voice); the cells marked '
           f'"itself, other salt" are the baseline: the full model\'s takes of one text at two different salts.\n\n')
    md += '### Charts\n\n'
    for name, alt in (('footprint', 'Disk size and peak device memory'), ('speed', 'Speed per arm'),
                      ('quality', 'Word error rate, failed takes and speaker similarity per arm')):
        md += f'![{alt}](charts/{name}.svg)\n\n'
    md += ('In the speed chart the real-time-factor and first-audio bars are the nearest-rank p50 rows of the table; '
           'decode and start-to-ready are arithmetic medians.\n\n')
    if 'grill' in t:
        md += grill_card(t)
    md += ('### How quality was scored\n\nWhisper large-v3 transcribed each take; reference and transcript went '
           'through Whisper\'s English normalizer before a word-level edit distance. Speaker embeddings are '
           f'SpeechBrain `spkrec-ecapa-voxceleb` at 16 kHz. Transcription ran on: '
           f'{", ".join(t["scoring"]["asr"])}.\n\n')
    md += '### Run history\n\n'
    md += f'- The run started at {E["started_at"]} with harness commit `{E["harness_commit"][:7]}`.\n'

    def listing(d):
        parts = [f'{n} under `{k}`' if k.endswith('/') else f'`{k}`' for k, n in d.items()]
        return ', '.join(parts[:-1]) + ' and ' + parts[-1] if len(parts) > 1 else ''.join(parts)

    for s in E['set_aside']:
        if 'refusal' in s:
            f = s['refusal']
            arm, salt = s['path'].split('/')[0], s['path'].rsplit('-s', 1)[1]
            md += (f'- The first pass stopped at {LABELS[arm]} salt {salt}: the benchmark client refused to start it, '
                   f'finding {f["missing"]} pinned files missing and {f["mismatched"]} mismatched. Missing: '
                   f'{listing(f["missing_under"])}'
                   + ('; all are hidden paths, which the client does not list' if f['missing_all_hidden'] else '')
                   + f'. {"No request was sent." if not s["requests_sent"] else "Requests had been sent."}\n')
    for r in E['resumed']:
        resumed_at = datetime.strptime(r['started_at'], '%Y-%m-%dT%H:%M:%S%z').timestamp()
        kept = [a for a in A if max(last for _, last in A[a]['request_epoch_span_per_salt'].values()) < resumed_at]
        for name, c in r['pin_changes'].items():
            if not c.get('pin_files_found'):
                md += f'- The {name} checkpoint\'s pins changed at the resume; the pin files are not in the run.\n'
                continue
            md += (f'- Pin change (harness commit `{r["harness_commit"][:7]}`): the {name} checkpoint\'s sha256sum '
                   f'list '
                   f'went from {c["entries_first"]} to {c["entries_resume"]} entries. Dropped: '
                   f'{listing(c["dropped_under"])}; added: {c["entries_added"]}; every kept entry has '
                   f'{"the same" if c["kept_entries_same_sha256"] else "NOT the same"} sha256 as before. The resumed '
                   f'preflight verified the {name} checkpoint\'s {r["checkpoint_files_verified"][name]} files '
                   f'(exit {r["checkpoint_verify_returncode"][name]}).')
            if 'client_pins' in c:
                p = c['client_pins']
                md += (f' The benchmark client got the same list less hidden paths: {p["entries"]} entries '
                       f'({listing(p["resume_entries_left_out"])} left out), '
                       f'{"all" if p["same_sha256_as_resume_pins"] else "NOT all"} with the same sha256.')
            md += '\n'
        same = r['same_as_first']
        aside = ', '.join(f'`{x["path"]}`' for x in E['set_aside'])
        md += (f'- The run resumed at {r["started_at"]}. Against the first preflight: '
               + ', '.join(f'{k.replace("_sha256", "")} {"same" if v else "DIFFERENT"}' for k, v in same.items())
               + '. Arms measured entirely before the resume, and kept: '
               + f'{", ".join(LABELS[a] for a in kept) or "none"}. '
               f'The set-aside directory ({aside}) is in no number here.\n')
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run', type=Path, help='run.py run directory, scored')
    ap.add_argument('out', type=Path, help='output directory')
    ap.add_argument('--grill', type=Path, help='`grill-perf media compare` JSON for the full-against-quant speed row')
    ap.add_argument('--grill-collector', help='collector name and version, e.g. "grill-perf 0.11.0"')
    ap.add_argument('--grill-collector-sha256', help='sha256 of the collector binary')
    ap.add_argument('--grill-arms', help='run-01 arm whose configuration each slot served, e.g. a=FG,b=Q,a2=FG')
    args = ap.parse_args()
    grill_opts = (args.grill_collector, args.grill_collector_sha256, args.grill_arms)
    if args.grill and not all(grill_opts):
        ap.error('--grill needs --grill-collector, --grill-collector-sha256 and --grill-arms')
    if not args.grill and any(grill_opts):
        ap.error('--grill-collector, --grill-collector-sha256 and --grill-arms need --grill')
    run, out = args.run, args.out
    preflight = json.loads((run / 'preflight.json').read_text())
    arms = [a for a in ('F', 'Q', 'FG', 'FB', 'QB') if (run / a / 's0' / 'whole' / 'summary.json').exists()]
    codec = {}
    for k, v in preflight['checkpoints'].items():
        path = pin_file(run, v['verify'])
        codec[k] = pins(path).get('codec.pth') if path else None
    tables = {'schema': 'full-vs-quant/1', 'arms': {}, 'environment': {
        'hardware': {k: v for k, v in preflight['hardware'].items() if k != 'host'},
        'engine_build': {k: preflight['binary_build'][k]
                         for k in ('commit', 'describe', 'branch', 'clean_tree', 'built', 'change')},
        'engine_binary_sha256': preflight['binary_sha256'], 'corpus_sha256': preflight['corpus_sha256'],
        'voice_sha256': preflight['voice_sha256'], 'harness_commit': preflight['harness_commit'],
        'checkpoint_pins_sha256': {k: v['verify']['pins_sha256'] for k, v in preflight['checkpoints'].items()},
        'checkpoint_files_verified': {k: v['verify']['files'] for k, v in preflight['checkpoints'].items()},
        'codec_pth_sha256': codec,
        'started_at': preflight['started_at'], 'salts': preflight['salts'],
        'resumed': [resume_record(run, preflight, json.loads(p.read_text()))
                    for p in sorted(run.glob('resume-*-preflight.json'))],
        'set_aside': set_aside(run)}}
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    requests, engine = [], []
    for arm in arms:
        rec, raw, eng = arm_record(run, arm, preflight)
        tables['arms'][arm] = rec
        requests += raw
        engine += eng
    scores, pairs = jl(run / 'scores.jsonl'), jl(run / 'pairs.jsonl')
    tables['scoring'] = {'asr': sorted({r.get('asr', 'not recorded') for r in scores})}
    finishes = {(r['arm'], r['salt'], r['item']): r['finish'] for r in requests
                if r['run'] == 'whole' and r['phase'] == 'warm' and r['round'] == 0}
    tables['quality'], tables['paired'] = quality(scores, pairs, arms, finishes)
    if args.grill:
        compare = args.grill.read_bytes()
        slot_arms = dict(x.split('=', 1) for x in args.grill_arms.split(','))
        if not set(slot_arms) == {'a', 'b', 'a2'} or not set(slot_arms.values()) <= set(arms):
            ap.error(f'--grill-arms must map a, b and a2 to arms of this run ({", ".join(arms)})')
        tables['grill'] = grill_record(compare, args.grill_collector, args.grill_collector_sha256, slot_arms)
        (out / 'raw' / 'grill-compare.json').write_bytes(compare)
    (out / 'raw' / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
    (out / 'raw' / 'engine.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in engine))
    (out / 'raw' / 'scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in scores))
    (out / 'raw' / 'speaker-pairs.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in pairs))
    (out / 'tables.json').write_text(json.dumps(tables, indent=1) + '\n')
    (out / 'tables.md').write_text(markdown(tables))
    (out / 'card-section.md').write_text(card(tables))
    charts(tables, out / 'charts')
    print(markdown(tables))


if __name__ == '__main__':
    main()
