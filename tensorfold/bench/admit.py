#!/usr/bin/env python3
"""Pre-load admission for one engine on a GB10: POSIX_FADV_DONTNEED on every regular file
under the run's own root (symlinks never followed; never a global drop), then admission only if both CUDA free
(cuMemGetInfo in a short driver-API subprocess, nothing allocated) and MemFree cover the stated peak plus margin.
Writes one JSON record (new file). Exit 0 admitted, 3 refused."""
import argparse
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

CUDA_QUERY = r"""
import ctypes, json
cuda = ctypes.CDLL("libcuda.so.1")
device, context, free, total = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_size_t(), ctypes.c_size_t()
out = {"free": None, "total": None}
for name, call in (("cuInit", lambda: cuda.cuInit(0)), ("cuDeviceGet", lambda: cuda.cuDeviceGet(ctypes.byref(device), 0)),
                   ("cuDevicePrimaryCtxRetain", lambda: cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device)),
                   ("cuCtxSetCurrent", lambda: cuda.cuCtxSetCurrent(context)),
                   ("cuMemGetInfo_v2", lambda: cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)))):
    rc = call()
    if rc != 0:
        out["error"] = f"{name} returned {rc}"
        break
else:
    out.update(free=free.value, total=total.value)
if context.value:
    cuda.cuDevicePrimaryCtxRelease_v2(device)
print(json.dumps(out))
"""


def meminfo() -> dict:
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemFree', 'MemAvailable', 'Cached'):
            fields[name] = int(value.split()[0]) * 1024
    return fields


def advise(root: Path) -> dict:
    """DONTNEED on each regular file under root; the walk never follows symlinks and each open refuses one."""
    record = {'root': str(root), 'files': 0, 'bytes': 0, 'errors': {}, 'memfree_before': meminfo()['MemFree']}
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = os.path.join(directory, name)
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError as error:
                if not os.path.islink(path):
                    record['errors'][path] = f'{type(error).__name__}: {error}'
                continue
            try:
                status = os.fstat(fd)
                if stat.S_ISREG(status.st_mode):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    record['files'] += 1
                    record['bytes'] += status.st_size
            except OSError as error:
                record['errors'][path] = f'{type(error).__name__}: {error}'
            finally:
                os.close(fd)
    record['memfree_after'] = meminfo()['MemFree']
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='the run-owned tree to advise, such as the checkpoint')
    parser.add_argument('--peak-bytes', type=int, required=True, help='expected engine peak device bytes')
    parser.add_argument('--peak-source', required=True, help='where the peak figure comes from')
    parser.add_argument('--margin-bytes', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    advice = advise(args.root.resolve())
    done = subprocess.run([sys.executable, '-B', '-c', CUDA_QUERY], capture_output=True, text=True, timeout=60,
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    cuda = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 and done.stdout.strip() else \
        {'free': None, 'error': done.stderr[-2000:]}
    memory = meminfo()
    need = args.peak_bytes + args.margin_bytes
    checks = {'cuda_free_covers_required': isinstance(cuda.get('free'), int) and cuda['free'] >= need,
              'memfree_covers_required': memory['MemFree'] >= need}
    record = {'schema': 'bench-admission/1', 'started_at_epoch': started, 'cache_advice': advice,
              'requirement': {'peak_bytes': args.peak_bytes, 'peak_source': args.peak_source,
                              'margin_bytes': args.margin_bytes, 'required_bytes': need},
              'cuda': cuda, 'meminfo': memory, 'checks': checks, 'admitted': all(checks.values()),
              'never': 'no global cache drop, no drop_caches, no file outside --root advised'}
    with open(args.out, 'x') as out:
        out.write(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'admitted': record['admitted'], 'required_gib': need / 2**30,
                      'cuda_free_gib': (cuda.get('free') or 0) / 2**30, 'memfree_gib': memory['MemFree'] / 2**30,
                      'advised_files': advice['files'], 'advice_errors': len(advice['errors'])}))
    return 0 if record['admitted'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
