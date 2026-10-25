#!/usr/bin/env python3
"""Quality scores for every warm take of a run.py run (whole replies, the first warm pass of each salt).

Per take:
- wer: Whisper large-v3 transcript against the corpus text, both through Whisper's English normalizer, word-level
  Levenshtein over the reference length;
- capped: the take reached the 1024-frame limit, so it did not end by itself;
- quiet_run_s: the longest run of 10 ms frames 40 dB or more below the take's 95th-percentile frame level, between
  the first and the last voiced frame (a stall inside speech);
- collapsed: wer > 0.2 or quiet_run_s >= 3.0 (G4b's collapse rule without its redraw clause: no redraw guard runs);
- failed: capped or collapsed;
- ECAPA speaker embeddings (SpeechBrain spkrec-ecapa-voxceleb, 16 kHz), from which pairwise cosines are computed:
  each arm's take against the full model's take of the same item and salt (arm F, or FB for the built-in voice),
  and, as the same-model baseline, the full model's takes of one item across salts.

Usage: score.py RUN_DIR --whisper DIR --ecapa DIR --speechbrain-path DIR   (writes RUN_DIR/scores.jsonl, pairs.jsonl)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio.functional as AF

CORPUS = Path(__file__).resolve().parents[1] / 'corpus' / 'drama-bench-en-v1.jsonl'
QUIET_DB, QUIET_RUN_S, WER_COLLAPSE, CAP_FRAMES, FRAME_SAMPLES = 40.0, 3.0, 0.2, 1024, 2048
BASELINE = {'F': 'F', 'Q': 'F', 'FG': 'F', 'QB': 'FB', 'FB': 'FB'}


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def quiet_run(y, rate):
    hop = rate // 100
    frames = y[:len(y) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(1)) + 1e-9)
    quiet = db < np.percentile(db, 95) - QUIET_DB
    voiced = np.flatnonzero(~quiet)
    if not len(voiced):
        return len(y) / rate
    longest = run = 0
    for q in quiet[voiced[0]:voiced[-1] + 1]:
        run = run + 1 if q else 0
        longest = max(longest, run)
    return longest / 100


def takes(run_dir):
    """(arm, salt, item, wav path, body sha256) for the first warm pass (round 0) of each salt's whole run."""
    out = []
    for arm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for salt_dir in sorted(arm_dir.glob('s*')):
            whole = salt_dir / 'whole'
            for line in (whole / 'samples.jsonl').read_text().splitlines():
                s = json.loads(line)
                if s['phase'] == 'warm' and s['round'] == 0:
                    out.append((arm_dir.name, int(salt_dir.name[1:]), s['item'],
                                whole / s['body_file'] if s.get('body_file') else None, s.get('body_sha256')))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--whisper', type=Path, required=True)
    ap.add_argument('--ecapa', type=Path, required=True)
    ap.add_argument('--speechbrain-path', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.speechbrain_path))
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    from transformers import pipeline

    texts = {json.loads(l)['id']: json.loads(l)['text'] for l in CORPUS.read_text().splitlines()}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device != 'cpu' else torch.float32
    asr = pipeline('automatic-speech-recognition', model=str(args.whisper), device=device, torch_dtype=dtype)
    normalize = asr.tokenizer.normalize
    ecapa = EncoderClassifier.from_hparams(source=str(args.ecapa), overrides={'pretrained_path': str(args.ecapa)},
                                           local_strategy=LocalStrategy.NO_LINK, run_opts={'device': device})
    rows, embeddings = [], {}
    for arm, salt, item, wav, body_sha in takes(args.run_dir):
        row = {'arm': arm, 'salt': salt, 'item': item, 'body_sha256': body_sha,
               'asr': f'{device.split(":")[0]} {str(dtype).removeprefix("torch.")}'}
        if wav is None:
            rows.append({**row, 'failed': True, 'missing': True})
            continue
        y, rate = soundfile.read(wav, dtype='float32')
        frames = len(y) // FRAME_SAMPLES
        y16 = AF.resample(torch.from_numpy(y), rate, 16000)
        hyp = asr({'raw': y16.numpy(), 'sampling_rate': 16000}, chunk_length_s=30, batch_size=4,
                  generate_kwargs={'language': 'english', 'task': 'transcribe'})['text']
        ref_words, hyp_words = normalize(texts[item]).split(), normalize(hyp).split()
        wer = edit_distance(ref_words, hyp_words) / len(ref_words)
        q = quiet_run(y, rate)
        capped = frames >= CAP_FRAMES
        collapsed = wer > WER_COLLAPSE or q >= QUIET_RUN_S
        with torch.inference_mode():
            e = ecapa.encode_batch(y16[None].to(device))[0, 0].float().cpu().numpy()
        embeddings[(arm, salt, item)] = e / np.linalg.norm(e)
        rows.append({**row, 'audio_s': round(len(y) / rate, 4), 'frames': frames, 'capped': capped,
                     'wer': round(wer, 5), 'ref_words': len(ref_words), 'hypothesis': hyp.strip(),
                     'quiet_run_s': q, 'collapsed': collapsed, 'failed': capped or collapsed})
        print(f'{arm} s{salt} {item}: wer {wer:.3f} quiet {q:.2f}s frames {frames}', flush=True)
    pairs = []
    for (arm, salt, item), e in embeddings.items():
        base = BASELINE[arm]
        if arm != base and (base, salt, item) in embeddings:
            pairs.append({'kind': 'arm_vs_full', 'arm': arm, 'baseline': base, 'salt': salt, 'item': item,
                          'cosine': float(e @ embeddings[(base, salt, item)])})
    for base in sorted(set(BASELINE.values())):
        items = sorted({i for a, _, i in embeddings if a == base})
        for item in items:
            salts = sorted(s for a, s, i in embeddings if a == base and i == item)
            for s1, s2 in itertools.combinations(salts, 2):
                pairs.append({'kind': 'full_across_salts', 'arm': base, 'baseline': base, 'salt': [s1, s2],
                              'item': item, 'cosine': float(embeddings[(base, s1, item)] @ embeddings[(base, s2, item)])})
    (args.run_dir / 'scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (args.run_dir / 'pairs.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in pairs))
    print(f'SCORED {len(rows)} takes, {len(pairs)} pairs', flush=True)


if __name__ == '__main__':
    main()
