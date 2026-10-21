#!/usr/bin/env python3
"""macOS port of recipe/pilot.py: the same guarded pilot, without systemd or cgroups.

Differences from the Linux launcher, and why:
- There is no user systemd and no cgroup, so the server runs as a direct child in its own
  process group and the launcher stops it with signals instead of `systemctl`.
- There is no /proc/meminfo. Admission and the abort threshold read `vm_stat`
  (free + inactive + speculative + purgeable pages). That is an approximation of what the
  kernel can hand out, not Linux's MemAvailable; macOS also counts GPU-wired memory as
  wired, so another model's resident weights are correctly excluded.
- TENSORFOLD_CUDA_MEMORY_LIMIT_GB does not exist on the Metal path, so the 54 GiB engine cap
  of the GB10 recipe has no macOS equivalent. Admission is weights + the 30 GiB headroom
  floor, and the abort threshold stays at 34 GiB available.
- `--min-available-gib` overrides the admission floor explicitly, for hosts already
  running another model. It is never silent: the figure used is written to session.json.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

GIB = 1024**3
HEADROOM = 30 * GIB
STOP_MARGIN = 4 * GIB
ABORT_AVAILABLE = HEADROOM + STOP_MARGIN


def page_size() -> int:
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    match = re.search(r'page size of (\d+) bytes', text)
    if not match:
        raise RuntimeError('vm_stat did not report a page size')
    return int(match.group(1))


def memory_available() -> int:
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    size = page_size()

    def pages(label: str) -> int:
        match = re.search(re.escape(label) + r':\s+(\d+)', text)
        return int(match.group(1)) if match else 0

    return size * sum(pages(label) for label in
                      ('Pages free', 'Pages inactive', 'Pages speculative', 'Pages purgeable'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=('native', 'sglang-omni'), default='native')
    parser.add_argument('--native-bin', type=Path, help='Patched tensorfold-native executable')
    parser.add_argument('--runtime-dir', type=Path, required=True,
                        help='Isolated runtime and cache directory')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True,
                        help='Existing lock file that every GPU workload on this host takes')
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--evidence-reuse', action='store_true',
                        help='Reuse an existing evidence directory (for a supervised service)')
    parser.add_argument('--port', type=int, default=18793)
    parser.add_argument('--host', default='0.0.0.0',
                        help='Bind address for the server; the port check uses it too')
    parser.add_argument('--ready-timeout', type=float, default=900)
    parser.add_argument('--min-available-gib', type=float, default=None,
                        help='Override the admission floor (default: weights + 30 GiB)')
    parser.add_argument('--reserve-gib', type=float, default=34,
                        help='TENSORFOLD_MEMORY_RESERVE_GIB passed to the server')
    parser.add_argument('serve_args', nargs='*', metavar='SERVE_ARG',
                        help='Extra tensorfold-native serve flags, after --')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if args.backend == 'sglang-omni':
        parser.error('sglang-omni needs the CUDA reference stack; this launcher is native-only')
    if args.native_bin is None:
        parser.error('The native backend requires --native-bin')

    root = args.runtime_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=args.evidence_reuse)
    executable = args.native_bin.resolve()
    weights = sum(p.stat().st_size for p in checkpoint.glob('*.safetensors'))
    weights += (checkpoint / 'codec.pth').stat().st_size
    if weights <= (checkpoint / 'codec.pth').stat().st_size:
        parser.error('No backbone safetensors found')
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f'Missing executable server: {executable}')
    if not args.gpu_lock.is_file():
        parser.error('The operator-specified shared GPU lock must already exist')

    server_command = [str(executable), 'serve', str(checkpoint), '--host', args.host,
                      '--port', str(args.port), '--name', 'fishaudio/drama-3',
                      '--context', '8192', '--parallel', '1', '--no-drafts',
                      '--prompt-cache-gib', '0', '--keep-warm', '0', *args.serve_args]

    with args.gpu_lock.open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another workload owns the GPU slot; no model was started')
        available = memory_available()
        floor = args.min_available_gib * GIB if args.min_available_gib is not None else weights + HEADROOM
        if available < floor:
            parser.error(f'Need {floor/GIB:.2f} GiB available, have {available/GIB:.2f}; no model was started')
        try:
            with socket.socket() as port_check:
                port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_check.bind((args.host, args.port))
        except OSError as exc:
            parser.error(f'Pilot port is unavailable: {exc}; no model was started')

        stop = threading.Event()
        failure: list[str] = []
        minimum_available = [available]
        started = time.monotonic()
        started_at_epoch = time.time()
        ready_at = None
        worker = None
        watcher = None

        environment = {
            'HOME': str(Path.home()),
            'PATH': f'{root / "venv/bin"}:/usr/bin:/bin:/usr/sbin:/sbin',
            'LANG': 'C.UTF-8',
            'PYTHONUNBUFFERED': '1',
            'HF_HOME': str(root / 'hf-cache'),
            'HF_HUB_OFFLINE': '1',
            'XDG_CACHE_HOME': str(root / 'runtime-cache'),
            'TENSORFOLD_MEMORY_RESERVE_GIB': f'{args.reserve_gib:g}',
            'TENSORFOLD_CUDA_MEMORY_LIMIT_GB': '54',
        }
        initial = {'backend': args.backend, 'weights_bytes': weights,
                   'initial_memavailable_bytes': available,
                   'required_memavailable_bytes': int(floor),
                   'headroom_floor_bytes': HEADROOM, 'abort_threshold_bytes': ABORT_AVAILABLE,
                   'memory_source': 'vm_stat free+inactive+speculative+purgeable',
                   'min_available_override_gib': args.min_available_gib,
                   'server_command': server_command, 'environment': environment,
                   'started_at_epoch': started_at_epoch,
                   'admission': 'weights + 30 GiB headroom (no cgroup on macOS)',
                   'allocator_caps_gib': {'engine_reserve_gib': args.reserve_gib},
                   'allocation_cap_excludes_driver_and_library_private_memory': True}
        (evidence / 'session.json').write_text(json.dumps(initial, indent=2) + '\n')
        (root / 'launched-at').write_text(f'{started_at_epoch:.6f}\n')
        print(json.dumps(initial), flush=True)

        def request_stop(signum: int, _frame: object) -> None:
            stop.set()

        def kill_tree(sig: int) -> None:
            if worker is not None and worker.poll() is None:
                try:
                    os.killpg(worker.pid, sig)
                except ProcessLookupError:
                    pass

        def watch_memory() -> None:
            try:
                with (evidence / 'memory.jsonl').open('w') as output:
                    while not stop.wait(0.1):
                        current = memory_available()
                        minimum_available[0] = min(minimum_available[0], current)
                        output.write(json.dumps({'elapsed_s': time.monotonic() - started,
                                                 'memavailable_bytes': current}) + '\n')
                        output.flush()
                        if current < ABORT_AVAILABLE:
                            raise MemoryError(
                                f'available memory {current/GIB:.2f} GiB crossed the '
                                f'{ABORT_AVAILABLE/GIB:.0f} GiB abort threshold')
            except Exception as exc:
                failure.append(f'{type(exc).__name__}: {exc}')
                stop.set()
                kill_tree(signal.SIGKILL)

        previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with (evidence / 'server.log').open('w') as server_log:
                worker = subprocess.Popen(server_command, stdout=server_log,
                                          stderr=subprocess.STDOUT, cwd=root,
                                          env=environment, start_new_session=True)
                watcher = threading.Thread(target=watch_memory, daemon=True)
                watcher.start()
                while not stop.is_set() and worker.poll() is None:
                    if ready_at is None:
                        if time.monotonic() - started > args.ready_timeout:
                            failure.append('Readiness deadline exceeded')
                            break
                        health_host = '127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host
                        try:
                            with urllib.request.urlopen(f'http://{health_host}:{args.port}/health',
                                                        timeout=1) as response:
                                if response.status == 200:
                                    ready_at = time.monotonic()
                                    initial['cold_ready_s'] = ready_at - started
                                    initial['ready_at_epoch'] = time.time()
                                    (evidence / 'ready.json').write_text(json.dumps(initial, indent=2) + '\n')
                                    print('DRAMA_PILOT_READY', flush=True)
                        except (urllib.error.URLError, TimeoutError):
                            pass
                    stop.wait(0.2)
                if worker.poll() not in (None, 0) and not stop.is_set():
                    failure.append(f'Server exited with status {worker.returncode}')
                elif worker.poll() is not None and ready_at is None and not stop.is_set():
                    failure.append('Server exited before readiness, despite exit status zero')
        except Exception as exc:
            failure.append(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            stop.set()
            if watcher is not None:
                watcher.join(timeout=3)
            kill_tree(signal.SIGTERM)
            if worker is not None:
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_tree(signal.SIGKILL)
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        failure.append('Server did not exit after SIGKILL')
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            final = {'ready': ready_at is not None, 'elapsed_s': time.monotonic() - started,
                     'minimum_memavailable_bytes': minimum_available[0],
                     'final_memavailable_bytes': memory_available(), 'failures': failure}
            (evidence / 'completion.json').write_text(json.dumps(final, indent=2) + '\n')
            print(json.dumps(final), flush=True)
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
