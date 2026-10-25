#!/usr/bin/env python3
"""Bounded guard for one oracle or reference-check run: the admission, memory cap, abort floor and shared lock of
recipe/pilot.py, plus a 600 s deadline. The command runs as a transient systemd --user unit in the runtime
directory, with that directory's venv/ and sglang-omni/ on its paths and every cache inside it."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SPEC = importlib.util.spec_from_file_location('drama_guard', Path(__file__).resolve().parents[1] / 'recipe/pilot.py')
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-lock', type=Path, required=True,
                        help='Existing lock file that every GPU workload on this host takes; never a second lock')
    parser.add_argument('--runtime-dir', type=Path, required=True, help='Holds venv/ and sglang-omni/; caches go here')
    parser.add_argument('evidence', type=Path, help='New evidence directory')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='The command to run, with its arguments')
    args = parser.parse_args()
    if not args.command:
        parser.error('a command is required')
    root = args.runtime_dir.resolve()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    if not args.gpu_lock.is_file():
        raise RuntimeError('Existing shared lock is missing')
    with args.gpu_lock.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = GUARD.memory_available()
        if available < 88 * GUARD.GIB:
            raise MemoryError(f'Admission requires 88 GiB available; have {available / GUARD.GIB:.2f}')
        unit = f'drama3-oracle-{os.getpid()}-{time.time_ns()}.service'
        command = ['systemd-run', '--user', f'--unit={unit}', '--wait', '--pipe', '--collect',
                   f'--property=MemoryMax={GUARD.CGROUP_CAP}', '--property=MemorySwapMax=0',
                   '--property=KillMode=control-group', '--property=OOMPolicy=kill',
                   '--property=TimeoutStopSec=10', '--property=RuntimeMaxSec=600',
                   f'--working-directory={root}', '/usr/bin/env', '-i',
                   f'HOME={Path.home()}', f'PATH={root / "venv/bin"}:{GUARD.CUDA_HOME / "bin"}:/usr/bin:/bin',
                   'LANG=C.UTF-8', 'PYTHONUNBUFFERED=1', 'CUDA_VISIBLE_DEVICES=0',
                   f'PYTHONPATH={root / "sglang-omni"}', 'HF_HUB_OFFLINE=1',
                   f'HF_HOME={root / "hf-cache"}', f'XDG_CACHE_HOME={root / "runtime-cache"}',
                   f'TRITON_CACHE_DIR={root / "runtime-cache/triton"}',
                   f'CUDA_CACHE_PATH={root / "runtime-cache/cuda"}',
                   f'LD_LIBRARY_PATH={root / "venv/lib/python3.12/site-packages/nvidia/cu13/lib"}',
                   *args.command]
        record = {'unit': unit, 'command': command, 'initial_memavailable_bytes': available,
                  'required_memavailable_bytes': 88 * GUARD.GIB,
                  'cap_bytes': GUARD.CGROUP_CAP, 'swap_cap_bytes': 0,
                  'abort_memavailable_bytes': 34 * GUARD.GIB, 'deadline_s': 600}
        (evidence / 'session.json').write_text(json.dumps(record, indent=2) + '\n')
        interrupted = []
        previous = {s: signal.signal(s, lambda sig, frame: interrupted.append(sig))
                    for s in (signal.SIGINT, signal.SIGTERM)}
        started = time.monotonic()
        minimum = available
        failures = []
        worker = None
        try:
            with (evidence / 'output.log').open('w') as log, (evidence / 'memory.jsonl').open('w') as memory:
                worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while worker.poll() is None:
                    current = GUARD.memory_available()
                    minimum = min(minimum, current)
                    memory.write(json.dumps({'elapsed_s': time.monotonic() - started,
                                             'memavailable_bytes': current}) + '\n')
                    memory.flush()
                    if current < 34 * GUARD.GIB or interrupted or time.monotonic() - started > 610:
                        failures.append('Memory abort, interruption or bounded deadline')
                        break
                    time.sleep(0.1)
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
        finally:
            # Keep ownership until systemd positively confirms the exact unit is stopped.
            while True:
                try:
                    subprocess.run(['systemctl', '--user', 'stop', unit], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                    state = subprocess.run(['systemctl', '--user', 'show', unit,
                                            '--property=LoadState', '--property=ActiveState'],
                                           capture_output=True, text=True, check=False, timeout=10)
                    props = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
                    if (props.get('ActiveState') in ('inactive', 'failed')
                            and (state.returncode == 0 or props.get('LoadState') == 'not-found')):
                        break
                except (OSError, subprocess.TimeoutExpired):
                    pass
                try:
                    subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all',
                                    '--signal=SIGKILL', unit], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                print(f'Retaining GPU lock: shutdown of {unit} not confirmed', flush=True)
                time.sleep(1)
            if worker is not None:
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            record.update({'returncode': None if worker is None else worker.returncode,
                           'elapsed_s': time.monotonic() - started,
                           'minimum_memavailable_bytes': minimum,
                           'final_memavailable_bytes': GUARD.memory_available(),
                           'stopped_unit_state': props, 'failures': failures})
            (evidence / 'completion.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record), flush=True)
        return 0 if worker is not None and worker.returncode == 0 and not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
