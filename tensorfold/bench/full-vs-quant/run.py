#!/usr/bin/env python3
"""Full model against the NVFP4 release quant: the GPU half of results/full-vs-quant-01.

One engine binary serves every arm; only the checkpoint and the guidance flags change. Each arm is started fresh four
times, once per sampling salt, through recipe/pilot.py, and measured with bench/drama_bench.py on the 24-item corpus:

  salt 0: whole replies (cold, 2 warmup, 3 warm passes), then streamed PCM on the same server (1 warmup, 3 passes)
  salts 1-3: whole replies (cold, 1 warm pass) on a fresh server each

Before each start, bench/admit.py drops the checkpoint's own pages from the page cache, so every load reads the disk.
Its admission verdict is recorded, not enforced: GB10 reports reclaimable page cache as used, and recipe/pilot.py's
MemAvailable gate plus the engine's own budget check are the guards. /health is sampled once a second while a server
is up. The run holds the host's GPU window locks for its whole length (Nemotron lock shared, slot A exclusive); each
pilot takes the Drama GPU lock itself.

Usage: run.py --out NEW_DIR --binary-dir DIR --full DIR --quant DIR --quant-sums FILE --voices DIR --runtime-dir DIR
              --library-dir DIR --gpu-lock FILE [--arms F,Q,FG,QB,FB] [--resume]

--resume continues an interrupted run in the existing --out with the same arguments: arm/salt pairs recorded in
status.json are kept untouched, a started but unrecorded pair's directory and admit record are moved aside, and
every preflight check is repeated into resume-N-preflight.json, which must agree with preflight.json.
"""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / 'bench'
CORPUS = BENCH / 'corpus' / 'drama-bench-en-v1.jsonl'
FULL_PINS = REPO / 'recipe' / 'checksums' / 'checkpoint.sha256'
PORT = 18797
SALTS = (0, 1, 2, 3)
GIB = 2 ** 30
VOICE = 'eq-3ref'  # three model-invented reference clips; requests send "default", which resolves to it
GUIDED = ('--cfg-gamma', '1.5', '--cfg-batched')
# arm: (checkpoint, reference voice, extra server flags, stated precision, expected /health peak for the admit record)
ARMS = {
    'F': ('full', True, (), 'FP8 checkpoint as distributed, unguided', 40 * GIB),
    'Q': ('quant', True, GUIDED, 'NVFP4 routed experts (GPTQ), 7,369 of 10,240 kept; guidance 1.5', 21 * GIB),
    'FG': ('full', True, GUIDED, 'FP8 checkpoint as distributed; guidance 1.5', 41 * GIB),
    'QB': ('quant', False, GUIDED, 'NVFP4 quant, built-in voice (guidance inactive without a reference)', 21 * GIB),
    'FB': ('full', False, (), 'FP8 checkpoint, built-in voice', 40 * GIB),
}
HF_CACHE = '.cache/huggingface/'  # `hf download --local-dir` bookkeeping (locks, download metadata), not model files
LOCKS = (('/tmp/nemo-gpu.lock', fcntl.LOCK_SH), ('/tmp/nemo-gpu.lock.slotA', fcntl.LOCK_EX))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def verify(pins, root):
    """sha256sum -c of every pinned file; returns the record, raises on any mismatch."""
    done = sh('sha256sum', '-c', '--strict', str(pins), cwd=root, timeout=1800)
    record = {'pins': str(pins), 'pins_sha256': sha256(pins), 'root': str(root), 'returncode': done.returncode,
              'files': len(done.stdout.splitlines()), 'failed': [l for l in done.stdout.splitlines() if not l.endswith(': OK')]}
    if done.returncode:
        raise SystemExit(f'checksum mismatch under {root}: {record["failed"][:5]} {done.stderr[-400:]}')
    return record


def quant_pins(out, sums, prefix):
    """Pins for the quant from the release SHA256SUMS, whose model/ lines are relative to the release folder.

    Returns (all, visible): every model file except the HF_CACHE directory, for sha256sum -c; and the same less
    hidden paths, for drama_bench, which lists only visible files and would read a hidden pin (.gitattributes) as
    missing. Hidden files stay covered by the first."""
    model = [(h, p[len('./model/'):]) for h, p in (l.split(None, 1) for l in sums.read_text().splitlines())
             if p.startswith('./model/') and not p.startswith('./model/' + HF_CACHE)]
    every, visible = out / f'{prefix}quant.sha256', out / f'{prefix}quant-visible.sha256'
    every.write_text(''.join(f'{h}  {p}\n' for h, p in model))
    visible.write_text(''.join(f'{h}  {p}\n' for h, p in model if not any(s.startswith('.') for s in p.split('/'))))
    return every, visible


def hardware():
    smi = sh('nvidia-smi', '--query-gpu=name,driver_version,pci.bus_id,compute_cap,power.limit', '--format=csv,noheader')
    cpu = sorted({l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                  if l.startswith(('model name', 'CPU part'))})
    mem = next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal'))
    nvcc = Path('/usr/local/cuda/bin/nvcc')
    cuda = sh(str(nvcc), '--version').stdout.strip().splitlines()[-1:] if nvcc.exists() else []
    return {'host': platform.node(), 'kernel': platform.release(), 'machine': platform.machine(),
            'gpu': smi.stdout.strip(), 'cpu': cpu, 'cpus': os.cpu_count(), 'memtotal': mem.split(':')[1].strip(),
            'nvcc': cuda, 'python': platform.python_version()}


def tree_bytes(root):
    files = {p.name: p.stat().st_size for p in sorted(root.iterdir()) if p.is_file() and not p.name.startswith('.')}
    return {'files': files, 'total_bytes': sum(files.values()),
            'weights_and_codec_bytes': sum(v for k, v in files.items() if k.endswith('.safetensors') or k == 'codec.pth')}


class Poller(threading.Thread):
    """/health once a second into a JSONL file until stopped."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.halt = path, threading.Event()

    def run(self):
        with open(self.path, 'a') as out:
            while not self.halt.is_set():
                t = time.time()
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                        out.write(json.dumps({'t': t, 'health': json.load(r)}) + '\n')
                except Exception as exc:  # the server is starting or stopping
                    out.write(json.dumps({'t': t, 'error': type(exc).__name__}) + '\n')
                out.flush()
                self.halt.wait(1.0)


def start(args, run, ckpt, voice, flags, salt):
    run.mkdir(parents=True)
    wrapper = run / 'server.sh'
    env = f'export TENSORFOLD_SEED_SALT={salt}\n' if salt else ''
    extra = (f' --voices {args.voices} --default-voice {VOICE}' if voice else '') + ' --max-frames 1024'
    wrapper.write_text(f'#!/bin/sh\n{env}exec {args.binary_dir / "tensorfold-native"} "$@"{extra} {" ".join(flags)}\n')
    wrapper.chmod(0o755)
    out = open(run / 'pilot.out', 'w')
    pilot = subprocess.Popen([sys.executable, str(REPO / 'recipe' / 'pilot.py'), '--native-bin', str(wrapper),
                              '--runtime-dir', str(args.runtime_dir), '--checkpoint', str(ckpt),
                              '--gpu-lock', str(args.gpu_lock), '--evidence-dir', str(run / 'pilot'),
                              '--library-dir', str(args.library_dir), '--port', str(PORT)],
                             stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    began = time.monotonic()
    while 'DRAMA_PILOT_READY' not in (run / 'pilot.out').read_text():
        if pilot.poll() is not None or time.monotonic() - began > 900:
            stop(pilot)
            raise RuntimeError(f'{run}: pilot not ready: ' + (run / 'pilot.out').read_text()[-800:])
        time.sleep(0.5)
    unit = json.loads((run / 'pilot' / 'session.json').read_text())['unit']
    pid = int(sh('systemctl', '--user', 'show', '-p', 'MainPID', '--value', unit).stdout.strip())
    return pilot, pid


def stop(pilot):
    if pilot.poll() is None:
        pilot.send_signal(signal.SIGINT)
        try:
            pilot.wait(timeout=180)
        except subprocess.TimeoutExpired:
            raise SystemExit('pilot did not stop within 180 s; GPU lock retained; stop by hand')


def bench(args, run, name, ckpt, pins, pid, label, precision, start_kind, warmup, repeats, stream=False):
    cmd = [sys.executable, str(BENCH / 'drama_bench.py'), 'run', '--endpoint', f'http://127.0.0.1:{PORT}',
           '--model', 'fishaudio/drama-3', '--corpus', str(CORPUS), '--evidence-dir', str(run / name),
           '--label', label, '--precision', precision, '--server-pid', str(pid), '--server-source', str(REPO),
           '--checkpoint', str(ckpt), '--checkpoint-pins', str(pins), '--pilot-evidence', str(run / 'pilot'),
           '--config', str(run / 'server.sh'), '--config', str(args.binary_dir / 'BUILD.json'),
           '--server-start', start_kind, '--warmup', str(warmup), '--repeats', str(repeats), '--timeout', '120']
    if stream:
        cmd += ['--response-format', 'pcm', '--stream']
    with open(run / f'{name}.out', 'w') as out:
        code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    if code == 2:
        raise RuntimeError(f'{run / name}: drama_bench refused (exit 2); see {name}.out')
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('out', 'binary_dir', 'full', 'quant', 'quant_sums', 'voices', 'runtime_dir', 'library_dir', 'gpu_lock'):
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--arms', default='F,Q,FG,QB,FB')
    ap.add_argument('--resume', action='store_true', help='continue the interrupted run in the existing --out')
    args = ap.parse_args()
    arms = args.arms.split(',')
    if any(a not in ARMS for a in arms):
        ap.error(f'arms are {",".join(ARMS)}')
    for name in ('binary_dir', 'full', 'quant', 'voices', 'runtime_dir', 'library_dir', 'gpu_lock', 'quant_sums'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    elif not (args.out / 'preflight.json').is_file():
        raise SystemExit(f'{args.out} holds no preflight.json; nothing to resume')
    held = []
    for path, mode in LOCKS:
        f = open(path, 'a')
        try:
            fcntl.flock(f, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'{path} is held by another GPU job; nothing started')
        held.append(f)

    prefix = f'resume-{1 + len(list(args.out.glob("resume-*-preflight.json")))}-' if args.resume else ''
    quant_all, quant_visible = quant_pins(args.out, args.quant_sums, prefix)
    pins = {'full': FULL_PINS, 'quant': quant_all}
    bench_pins = {'full': FULL_PINS, 'quant': quant_visible}
    roots = {'full': args.full, 'quant': args.quant}
    bin_sums = args.binary_dir / 'SHA256SUMS'
    preflight = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'hardware': hardware(), 'argv': sys.argv,
                 'harness_commit': sh('git', '-C', str(REPO), 'rev-parse', 'HEAD').stdout.strip(),
                 'harness_dirty': bool(sh('git', '-C', str(REPO), 'status', '--porcelain').stdout.strip()),
                 'binary_build': json.loads((args.binary_dir / 'BUILD.json').read_text()),
                 'binary_sha256': sha256(args.binary_dir / 'tensorfold-native'),
                 'binary_sums': verify(bin_sums, args.binary_dir) if bin_sums.exists() else None,
                 'voice_sha256': sha256(args.voices / f'{VOICE}.json'), 'corpus_sha256': sha256(CORPUS),
                 'checkpoints': {k: {**tree_bytes(v), 'verify': verify(pins[k], v)} for k, v in roots.items()},
                 'arms': {a: dict(zip(('checkpoint', 'reference_voice', 'flags', 'precision', 'admit_peak_bytes'),
                                      ARMS[a])) for a in arms}, 'salts': SALTS, 'port': PORT}
    status = {}
    if args.resume:
        first = json.loads((args.out / 'preflight.json').read_text())
        again = json.loads(json.dumps(preflight))
        differs = [k for k in ('binary_sha256', 'binary_build', 'voice_sha256', 'corpus_sha256', 'salts', 'port')
                   if first[k] != again[k]]
        differs += [f'checkpoints.{k}.root' for k in roots
                    if first['checkpoints'][k]['verify']['root'] != again['checkpoints'][k]['verify']['root']]
        differs += [f'arms.{a}' for a in arms if a in first['arms'] and first['arms'][a] != again['arms'][a]]
        if differs:
            raise SystemExit(f'resume refused: {differs} differ from {args.out / "preflight.json"}; nothing started')
        if (args.out / 'status.json').is_file():
            status = json.loads((args.out / 'status.json').read_text())
        preflight['resumes'] = str(args.out / 'preflight.json')
    (args.out / f'{prefix}preflight.json').write_text(json.dumps(preflight, indent=1) + '\n')
    print(f'{prefix}preflight ok', flush=True)

    for arm in arms:
        kind, voice, flags, precision, peak = ARMS[arm]
        ckpt = roots[kind]
        for salt in SALTS:
            run = args.out / arm / f's{salt}'
            if f'{arm}/s{salt}' in status:
                print(f'{time.strftime("%T")} {arm} salt {salt}: kept from earlier, bench exits '
                      f'{status[f"{arm}/s{salt}"]}', flush=True)
                continue
            admit = args.out / f'admit-{arm}-s{salt}.json'
            for leftover in (run, admit):
                if args.resume and leftover.exists():
                    aside = leftover.with_name(f'{prefix}aside-{leftover.name}')
                    leftover.rename(aside)
                    print(f'{time.strftime("%T")} {arm} salt {salt}: unfinished {leftover} moved to {aside}', flush=True)
            sh(sys.executable, str(BENCH / 'admit.py'), '--root', str(ckpt), '--peak-bytes', str(peak),
               '--peak-source', 'expected /health peak (smoke on this host), advisory', '--margin-bytes', str(2 * GIB),
               '--out', str(admit))
            print(f'{time.strftime("%T")} {arm} salt {salt}: starting', flush=True)
            pilot, pid = start(args, run, ckpt, voice, flags, salt)
            poller = Poller(run / 'health.jsonl')
            poller.start()
            try:
                label = f'full-vs-quant-01 {arm} salt {salt}'
                if salt == 0:
                    codes = [bench(args, run, 'whole', ckpt, bench_pins[kind], pid, label, precision, 'fresh', 2, 3),
                             bench(args, run, 'stream', ckpt, bench_pins[kind], pid, label + ' stream', precision, 'reused',
                                   1, 3, stream=True)]
                else:
                    codes = [bench(args, run, 'whole', ckpt, bench_pins[kind], pid, label, precision, 'fresh', 0, 1)]
            finally:
                poller.halt.set()
                stop(pilot)
                poller.join(5)
            status[f'{arm}/s{salt}'] = codes
            print(f'{time.strftime("%T")} {arm} salt {salt}: bench exits {codes}', flush=True)
            (args.out / 'status.json').write_text(json.dumps(status, indent=1) + '\n')
    print('RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
