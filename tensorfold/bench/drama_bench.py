#!/usr/bin/env python3
"""Bounded benchmark client for POST /v1/audio/speech answering mono PCM16 as WAV or raw PCM.

run      sends a fixed corpus schedule one request at a time (cold, warmup and warm phases) and writes one
         evidence directory: server identities, every raw request sample, every complete 200 body, and
         per-phase summaries.
compare  pairs two complete runs' samples by (phase, round, item) over the same corpus, schedule and contract.

Timings are client end-to-end HTTP. Whole replies (the default) must use Content-Length framing. With --stream,
requests ask for raw PCM streamed as an HTTP/1.1 chunked body, decoded strictly, and the first whole codec frame of
received audio bytes is the first-playable-audio time. Every network exchange, identity step and accepted input
has an explicit time or size bound.
Standard library only.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import ipaddress
import json
import math
import os
import platform
import re
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

SPEECH_PATH = '/v1/audio/speech'
MEDIA_TYPES = {'wav': 'audio/wav', 'pcm': 'audio/pcm'}  # the native route's contentType()
MAX_INPUT_CHARS = 4096  # the native route's max_input_chars
BUILDER_TEXT = ('<|im_start|>', '<|im_end|>', '<|voice|>', '<|text|>', '<|interleave|>', '<|semantic:', '<|speaker')
ITEM_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}')
# Bounds on everything the client accepts or holds.
MAX_CORPUS_BYTES = 1 << 20
MAX_CORPUS_ITEMS = 1000
MAX_REQUESTS = 2000
MAX_TIMEOUT_S = 3600.0
MAX_RESPONSE_BYTES = 64 << 20
MAX_HEAD_BYTES = 64 << 10  # a response head, and separately a chunked body's trailer section
MAX_CHUNK_LINE = 1024  # one chunk-size line, extensions included
MAX_RECORD_BYTES = 16 << 20  # each metadata file, config, /proc map, git listing and run file read
MAX_LISTED_FILES = 10000  # checkpoint files and untracked source files
HASHED_FILE_LIMIT = 64 << 20  # larger checkpoint files are identified by pin, size and mtime, not rehashed
SNAPSHOT_BYTES = 1 << 20
SNAPSHOT_TIMEOUT_S = 10.0
IDENTITY_DEADLINE_S = 300.0
WAV_HEADER_SLACK = 4096
DETAIL_CHARS = 2048
PHASES = ('cold', 'warmup', 'warm')
TIMING_SCOPE = 'client end-to-end: connect, upload, server work and download to the last body byte'
NO_FIRST_PLAYABLE = 'requests never ask for streaming; only whole responses are timed'
FIRST_PLAYABLE = 'first whole codec frame of received audio bytes, from the request start (chunk boundaries ignored)'


class HarnessError(Exception):
    """A refusal: nothing is measured on top of an unverified identity, corpus or server."""


class AudioError(ValueError):
    """A 200 response whose body is not the contracted audio."""


class ResponseFault(Exception):
    """A response that cannot be accepted; ``outcome`` names its failure class."""

    def __init__(self, outcome: str, detail: str):
        super().__init__(detail)
        self.outcome = outcome


class Deadline:
    """An absolute monotonic deadline; ``left`` raises TimeoutError once it has passed."""

    def __init__(self, seconds: float, what: str):
        self.at = time.perf_counter() + seconds
        self.what = what

    def left(self) -> float:
        remaining = self.at - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f'{self.what} exceeded its deadline')
        return remaining


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, deadline: Deadline) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while deadline.left() and (block := stream.read(1 << 20)):
            digest.update(block)
    return digest.hexdigest()


def read_bounded(path: Path, limit: int | None = None) -> bytes:
    """The whole file, refused past ``limit`` (default MAX_RECORD_BYTES) without reading further."""
    limit = MAX_RECORD_BYTES if limit is None else limit
    with open(path, 'rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise HarnessError(f'{path} exceeds {limit} bytes')
    return data


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')


# Corpus and schedule

def load_corpus(path: Path) -> tuple[list[dict], str]:
    """JSONL of {"id", "text"} in send order, and the file's sha256."""
    raw = read_bounded(path, MAX_CORPUS_BYTES)
    items: list[dict] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        where = f'{path}:{number}'
        if len(items) == MAX_CORPUS_ITEMS:
            raise HarnessError(f'{where}: a corpus holds at most {MAX_CORPUS_ITEMS} items')
        try:
            item = json.loads(line)
        except ValueError as error:
            raise HarnessError(f'{where}: {error}') from None
        if not isinstance(item, dict) or set(item) != {'id', 'text'}:
            raise HarnessError(f'{where}: each line must be an object with exactly "id" and "text"')
        if not isinstance(item['id'], str) or not ITEM_ID.fullmatch(item['id']) or item['id'] in seen:
            raise HarnessError(f'{where}: ids must be unique [A-Za-z0-9._-] names of at most 64 characters')
        text = item['text']
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT_CHARS:
            raise HarnessError(f'{where}: text must be non-empty and at most {MAX_INPUT_CHARS} characters')
        if any(token in text for token in BUILDER_TEXT):
            raise HarnessError(f'{where}: text must not contain prompt-builder or speaker tags')
        seen.add(item['id'])
        items.append(item)
    if not items:
        raise HarnessError(f'{path}: the corpus is empty')
    return items, sha256_bytes(raw)


def plan(items: list[dict], fresh: bool, warmup: int, repeats: int) -> list[tuple[str, int, dict]]:
    """(phase, round, item) in send order; only a fresh process's first request is cold."""
    if warmup < 0 or repeats < 1 or fresh + warmup + repeats * len(items) > MAX_REQUESTS:
        raise HarnessError(f'need --warmup >= 0, --repeats >= 1 and at most {MAX_REQUESTS} scheduled requests')
    schedule = [('cold', 0, items[0])] if fresh else []
    schedule += [('warmup', i // len(items), items[i % len(items)]) for i in range(warmup)]
    schedule += [('warm', r, item) for r in range(repeats) for item in items]
    return schedule


def contract_of(args: argparse.Namespace) -> tuple[dict, int]:
    """The audio contract and its response-byte limit, refused past the client's bounds (NaN and inf included)."""
    if not 0 < args.timeout <= MAX_TIMEOUT_S:
        raise HarnessError(f'--timeout must be positive and at most {MAX_TIMEOUT_S} s')
    if min(args.sample_rate, args.frame_samples, args.max_frames) < 1:
        raise HarnessError('--sample-rate, --frame-samples and --max-frames must be positive')
    if args.stream and args.response_format != 'pcm':
        raise HarnessError('--stream sends raw PCM: use --response-format pcm')
    limit = WAV_HEADER_SLACK + 2 * args.max_frames * args.frame_samples
    if limit > MAX_RESPONSE_BYTES:
        raise HarnessError(f'--max-frames x --frame-samples allows {limit} response bytes; the bound is {MAX_RESPONSE_BYTES}')
    return {'response_format': args.response_format, 'stream': args.stream, 'sample_rate': args.sample_rate,
            'frame_samples': args.frame_samples, 'max_frames': args.max_frames, 'voice': 'default'}, limit


# HTTP

class Endpoint:
    """An http:// base URL whose host is an IP literal, so no name resolution sits outside a deadline."""

    def __init__(self, url: str):
        parts = urlsplit(url)
        try:
            address = ipaddress.ip_address(parts.hostname or '')
            port = parts.port or 80
        except ValueError:
            address = None
        if parts.scheme != 'http' or address is None or parts.path.strip('/') or parts.query:
            raise HarnessError('--endpoint must be a bare http://IP[:port] base URL')
        self.url = url.rstrip('/')
        self.authority = parts.netloc
        self.family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        self.address = (str(address), port)


def parse_head(head: bytes) -> tuple[int, dict[str, str]]:
    lines = head.decode('iso-8859-1').split('\r\n')
    status = lines[0].split(' ', 2)
    if len(status) < 2 or not status[0].startswith('HTTP/') or not status[1].isdigit():
        raise ResponseFault('protocol', f'bad status line {lines[0][:80]!r}')
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, colon, value = line.partition(':')
        name = name.strip().lower()
        if not colon or name in headers:
            raise ResponseFault('protocol', f'bad or repeated header line {line[:80]!r}')
        headers[name] = value.strip()
    return int(status[1]), headers


TOKEN = rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
QUOTED = rb'"(?:[\t !#-\[\]-~\x80-\xff]|\\[\t -~\x80-\xff])*"'
CHUNK_SIZE_LINE = re.compile(rb'([0-9A-Fa-f]{1,8})(?:[ \t]*;[ \t]*' + TOKEN + rb'(?:[ \t]*=[ \t]*(?:' + TOKEN + rb'|'
                             + QUOTED + rb'))?)*')
TRAILER_FIELD = re.compile(TOKEN + rb':[\t \x21-\x7e\x80-\xff]*')


class ChunkedDecoder:
    """An incremental HTTP/1.1 chunked-body decoder for the RFC 9112 section 7.1 grammar, with bounds.

    Accepted: chunk-size = 1-8 hex digits (RFC: 1*HEXDIG; bounded here), then chunk-ext = *(BWS ";" BWS token
    [BWS "=" BWS (token / quoted-string)]), then CRLF; chunk data followed by exactly CRLF; after the zero chunk a
    trailer section of field lines (token ":" field-value, no control bytes but HTAB) each ending in CRLF, then the
    empty line. Bounds: a chunk-size line holds at most MAX_CHUNK_LINE bytes before its CRLF; the whole trailer
    section, every CRLF and the final empty line included, at most MAX_HEAD_BYTES. A bound is judged the same
    however the input is split, a lone CR still waiting for its LF included. Anything else is ``protocol`` as soon
    as it is visible: a CR or LF outside a CRLF, a bad line, wrong bytes after chunk data, any byte after the final
    empty line. Decoded data past ``limit`` is ``oversized_body``. Every loop step
    consumes at least one byte or returns, so feed always makes progress."""

    def __init__(self, limit: int):
        self.limit = limit
        self.pending = bytearray()
        self.data = bytearray()
        self.state = 'size'
        self.left = 0
        self.trailer_bytes = 0
        self.done = False

    def line(self, cap: int, what: str) -> bytes | None:
        """The next line without its CRLF, or None while it is incomplete. Refused at once: a CR or LF that is not
        part of a CRLF (a trailing CR still awaiting its LF waits), or content past ``cap`` (that CR not counted)."""
        end = self.pending.find(b'\r\n')
        first = min((i for i in (self.pending.find(b'\r'), self.pending.find(b'\n')) if i >= 0), default=-1)
        awaiting_lf = first == len(self.pending) - 1 and self.pending[first] == 13
        if first >= 0 and first != end and not awaiting_lf:
            raise ResponseFault('protocol', f'bare CR or LF in a {what}')
        content = end if end >= 0 else len(self.pending) - self.pending.endswith(b'\r')
        if content > cap:
            raise ResponseFault('protocol', f'{what} exceeds {max(cap, 0)} bytes')
        if end < 0:
            return None
        text = bytes(self.pending[:end])
        del self.pending[:end + 2]
        return text

    def feed(self, received: bytes) -> None:
        self.pending += received
        while self.pending and not self.done:
            if self.state == 'size':
                text = self.line(MAX_CHUNK_LINE, 'chunk-size line')
                if text is None:
                    return
                size = CHUNK_SIZE_LINE.fullmatch(text)
                if size is None:
                    raise ResponseFault('protocol', f'bad chunk-size line {text[:40]!r}')
                self.left = int(size[1], 16)
                if len(self.data) + self.left > self.limit:
                    raise ResponseFault('oversized_body', f'chunked body exceeds {self.limit} bytes')
                self.state = 'data' if self.left else 'trailer'
            elif self.state == 'data':
                take = min(self.left, len(self.pending))
                self.data += self.pending[:take]
                del self.pending[:take]
                self.left -= take
                if not self.left:
                    self.state = 'crlf'
            elif self.state == 'crlf':
                if self.pending[:2] != b'\r\n'[:len(self.pending[:2])]:
                    raise ResponseFault('protocol', 'chunk data not followed by CRLF')
                if len(self.pending) < 2:
                    return
                del self.pending[:2]
                self.state = 'size'
            else:  # trailer section: field lines, then the empty line that ends the body
                text = self.line(MAX_HEAD_BYTES - self.trailer_bytes - 2, 'trailer section')
                if text is None:
                    return
                self.trailer_bytes += len(text) + 2
                if not text:
                    self.done = True
                elif not TRAILER_FIELD.fullmatch(text):
                    raise ResponseFault('protocol', f'bad trailer field line {text[:40]!r}')
        if self.done and self.pending:
            raise ResponseFault('protocol', 'bytes after the terminating chunk')


def exchange(endpoint: Endpoint, method: str, path: str, body: bytes, timeout: float, limit: int,
             frame_bytes: int | None = None) -> tuple[dict, bytes | None]:
    """One request on a new connection under one absolute deadline covering connect, upload, head and every
    body read. A whole reply must use Content-Length framing, which every native whole reply uses. With
    ``frame_bytes`` a 200 must be a chunked stream without Content-Length; its decoded bytes are the body, and the
    arrival of its first byte and first ``frame_bytes`` are timed from the audio bytes, not chunk boundaries. A
    transport or protocol fault becomes the outcome; the body is None then."""
    result = {'outcome': 'ok', 'detail': None, 'head_s': None, 'total_s': None, 'status': None,
              'content_type': None, 'content_length': None, 'body_bytes': 0,
              'first_audio_byte_s': None, 'first_frame_s': None}
    began = time.perf_counter()
    deadline = Deadline(timeout, f'the {timeout} s request')
    head = f'{method} {path} HTTP/1.1\r\nHost: {endpoint.authority}\r\nConnection: close\r\n'
    if body:
        head += 'Content-Type: application/json\r\n'
    request = (head + f'Content-Length: {len(body)}\r\n\r\n').encode() + body
    received = bytearray()
    sock = socket.socket(endpoint.family, socket.SOCK_STREAM)

    def fill() -> bool:
        """One recv bounded by the time left; False at end of stream."""
        sock.settimeout(deadline.left())
        chunk = sock.recv(1 << 16)
        received.extend(chunk)
        return bool(chunk)

    try:
        sock.settimeout(deadline.left())
        sock.connect(endpoint.address)
        sock.settimeout(deadline.left())
        sock.sendall(request)  # one timeout bounds the whole sendall
        while (end := received.find(b'\r\n\r\n')) < 0:
            if len(received) > MAX_HEAD_BYTES:
                raise ResponseFault('protocol', f'response head exceeds {MAX_HEAD_BYTES} bytes')
            if not fill():
                raise ResponseFault('protocol', 'connection closed before a complete response head')
        result['head_s'] = time.perf_counter() - began
        status, headers = parse_head(bytes(received[:end]))
        result.update(status=status, content_type=headers.get('content-type'))
        del received[:end + 4]
        if frame_bytes and status == 200:
            if headers.get('transfer-encoding', '').lower() != 'chunked' or 'content-length' in headers:
                raise ResponseFault('protocol', 'a streamed reply must be chunked and carry no Content-Length')
            decoder = ChunkedDecoder(limit)
            while True:
                decoder.feed(bytes(received))
                received.clear()
                now, have = time.perf_counter() - began, len(decoder.data)
                if have and result['first_audio_byte_s'] is None:
                    result['first_audio_byte_s'] = now
                if have >= frame_bytes and result['first_frame_s'] is None:
                    result['first_frame_s'] = now
                result['body_bytes'] = have
                if decoder.done:
                    return result, bytes(decoder.data)
                if not fill():
                    raise ResponseFault('incomplete_body', f'the stream ended without its terminating chunk '
                                                           f'after {have} bytes')
        if 'transfer-encoding' in headers:
            raise ResponseFault('protocol', f'Transfer-Encoding {headers["transfer-encoding"][:40]!r} is not '
                                            'the native Content-Length framing of a whole reply')
        length = headers.get('content-length', '')
        if not (length.isascii() and length.isdigit()):
            raise ResponseFault('protocol', f'missing or bad Content-Length {length[:40]!r}')
        declared = result['content_length'] = int(length)
        if declared > limit:
            raise ResponseFault('oversized_body', f'Content-Length {declared} exceeds {limit} bytes')
        while len(received) < declared and fill():  # each fill consumes bytes or ends the stream
            pass
        data = bytes(received[:declared])
        result['body_bytes'] = len(data)
        if len(data) < declared:
            raise ResponseFault('incomplete_body', f'{len(data)} of {declared} declared bytes')
        return result, data
    except TimeoutError as error:
        result.update(outcome='timeout', detail=str(error))
    except ResponseFault as fault:
        result.update(outcome=fault.outcome, detail=str(fault))
    except OSError as error:
        result.update(outcome='connection', detail=f'{type(error).__name__}: {error}')
    finally:
        result['total_s'] = time.perf_counter() - began
        sock.close()
    return result, None


def snapshot(endpoint: Endpoint, path: str, timeout: float = SNAPSHOT_TIMEOUT_S) -> dict:
    """A bounded GET kept as evidence; a failure is recorded, not raised."""
    result, data = exchange(endpoint, 'GET', path, b'', timeout, SNAPSHOT_BYTES)
    if data is None:
        return {'path': path, 'error': f'{result["outcome"]}: {result["detail"]}'}
    return {'path': path, 'status': result['status'], 'content_type': result['content_type'],
            'body': data.decode('utf-8', 'replace')}


def server_state(endpoint: Endpoint) -> dict:
    """Service-side views, kept raw and never merged with client timings."""
    return {'at_epoch': time.time(), **{name: snapshot(endpoint, path) for name, path in
                                        (('health', '/health'), ('models', '/v1/models'), ('metrics', '/metrics'))}}


def error_message(body: bytes) -> str:
    try:
        return str(json.loads(body)['error']['message'])[:DETAIL_CHARS]
    except (ValueError, KeyError, TypeError):
        return body[:DETAIL_CHARS].decode('utf-8', 'replace')


# Audio

def parse_wav(data: bytes, sample_rate: int) -> bytes:
    """The data chunk of a mono 16-bit PCM RIFF/WAVE body at ``sample_rate``."""
    if len(data) < 12 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise AudioError('not a RIFF/WAVE body')
    if int.from_bytes(data[4:8], 'little') != len(data) - 8:
        raise AudioError('RIFF size does not match the body length')
    chunks: dict[bytes, bytes] = {}
    at = 12
    while at < len(data):
        if at + 8 > len(data):
            raise AudioError('truncated chunk header')
        kind, size = data[at:at + 4], int.from_bytes(data[at + 4:at + 8], 'little')
        end = at + 8 + size
        if end > len(data):
            raise AudioError(f'{kind!r} chunk overruns the body')
        if kind in chunks:
            raise AudioError(f'duplicate {kind!r} chunk')
        chunks[kind] = data[at + 8:end]
        at = end + (size & 1)
    if b'fmt ' not in chunks or b'data' not in chunks or len(chunks[b'fmt ']) < 16:
        raise AudioError('missing fmt or data chunk')
    tag, channels, rate, byte_rate, align, bits = struct.unpack('<HHIIHH', chunks[b'fmt '][:16])
    if (tag, channels, bits) != (1, 1, 16):
        raise AudioError(f'expected PCM mono 16-bit, got format {tag}, {channels} channels, {bits} bits')
    if rate != sample_rate:
        raise AudioError(f'sample rate {rate} is not the contracted {sample_rate}')
    if byte_rate != 2 * rate or align != 2:
        raise AudioError('byte rate or block align disagrees with mono 16-bit')
    return chunks[b'data']


def measure_pcm(pcm: bytes, sample_rate: int, frame_samples: int, max_frames: int) -> dict:
    """Sample-count checks and level statistics of PCM16LE mono."""
    if not pcm or len(pcm) % 2:
        raise AudioError(f'{len(pcm)} PCM bytes are not a non-zero whole number of 16-bit samples')
    samples = len(pcm) // 2
    frames, partial = divmod(samples, frame_samples)
    if partial:
        raise AudioError(f'{samples} samples are not a whole number of {frame_samples}-sample frames')
    if frames > max_frames:
        raise AudioError(f'{frames} frames exceed the {max_frames}-frame limit')
    values = array.array('h')
    values.frombytes(pcm)
    if sys.byteorder == 'big':
        values.byteswap()
    power = sum(v * v for v in values) / samples
    return {'samples': samples, 'frames': frames, 'duration_s': samples / sample_rate,
            'peak': max(max(values), -min(values)),
            'clipped_samples': values.count(32767) + values.count(-32767) + values.count(-32768),
            'rms_dbfs': 10 * math.log10(power / 32767 ** 2) if power else None}


def audio_of(data: bytes, content_type: str | None, contract: dict) -> dict:
    """Raw PCM has no header, so its media type is the only proof of its encoding: it must be the requested one."""
    media = (content_type or '').split(';')[0].strip().lower()
    wanted = MEDIA_TYPES[contract['response_format']]
    if media != wanted:
        raise AudioError(f'media type {content_type!r} is not the requested {wanted}')
    pcm = parse_wav(data, contract['sample_rate']) if contract['response_format'] == 'wav' else data
    return measure_pcm(pcm, contract['sample_rate'], contract['frame_samples'], contract['max_frames'])


# Identities. Every step checks one IDENTITY_DEADLINE_S deadline between filesystem operations and git is killed
# at it; a single blocking local syscall (open, stat, read on a hung filesystem) cannot be interrupted.

def process_start_ticks(pid: int) -> int:
    return int(read_bounded(Path('/proc', str(pid), 'stat')).rsplit(b')', 1)[1].split()[19])


def process_identity(pid: int, deadline: Deadline) -> dict:
    """The running server: executable bytes, argv and mapped files (by size and mtime, not content)."""
    proc = Path('/proc', str(pid))
    try:
        deadline.left()
        start = process_start_ticks(pid)
        exe = os.readlink(proc / 'exe')
        exe_sha256 = sha256_file(proc / 'exe', deadline)
        deadline.left()
        cmdline = [arg.decode('utf-8', 'replace') for arg in read_bounded(proc / 'cmdline').split(b'\0')[:-1]]
        deadline.left()
        maps = read_bounded(proc / 'maps').decode('utf-8', 'replace')
    except OSError as error:
        raise HarnessError(f'cannot inspect server process {pid}: {error}') from None
    mapped = sorted({fields[5] for fields in (line.split(None, 5) for line in maps.splitlines())
                     if len(fields) == 6 and fields[5].startswith('/') and not fields[5].startswith('/dev/')})
    files = []
    for path in mapped:
        deadline.left()
        try:
            status = os.stat(path)
            files.append({'path': path, 'bytes': status.st_size, 'mtime_ns': status.st_mtime_ns})
        except OSError:
            files.append({'path': path, 'missing': True})
    return {'pid': pid, 'start_ticks': start, 'exe': exe, 'exe_sha256': exe_sha256, 'cmdline': cmdline,
            'mapped_files': files, 'mapped_files_content_hashed': False}


def git(repo: Path, deadline: Deadline, *args: str, sink=None, check: bool = True) -> bytes:
    """A read-only git command killed at the deadline. Output streams to ``sink`` or, up to MAX_RECORD_BYTES,
    is returned."""
    kept = bytearray()

    def keep(block: bytes) -> None:
        kept.extend(block)
        if len(kept) > MAX_RECORD_BYTES:
            raise HarnessError(f'git {args[0]} output in {repo} exceeds {MAX_RECORD_BYTES} bytes')

    left = deadline.left()
    try:
        proc = subprocess.Popen(['git', '-C', str(repo), '--no-optional-locks', *args],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as error:
        raise HarnessError(f'cannot run git: {error}') from None
    timer = threading.Timer(left, proc.kill)
    timer.start()
    try:
        while block := proc.stdout.read(1 << 16):
            (sink or keep)(block)
        code = proc.wait()
    finally:
        timer.cancel()
        proc.kill()
        proc.wait()
        proc.stdout.close()
    deadline.left()
    if check and code != 0:
        raise HarnessError(f'git {" ".join(args)} failed in {repo} (exit {code})')
    return bytes(kept)


def source_identity(repo: Path, deadline: Deadline) -> dict:
    """HEAD alone does not identify a tree with uncommitted work: the tracked diff and untracked files are hashed too."""
    head = git(repo, deadline, 'rev-parse', 'HEAD').decode().strip()
    branch = git(repo, deadline, 'symbolic-ref', '-q', '--short', 'HEAD', check=False).decode().strip() or None
    diff, size = hashlib.sha256(), [0]

    def hash_diff(block: bytes) -> None:
        diff.update(block)
        size[0] += len(block)

    git(repo, deadline, 'diff', '--binary', 'HEAD', sink=hash_diff)
    untracked = [name.decode() for name in
                 git(repo, deadline, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if name]
    if len(untracked) > MAX_LISTED_FILES:
        raise HarnessError(f'{repo} has {len(untracked)} untracked files; the bound is {MAX_LISTED_FILES}')
    return {'path': str(repo), 'head': head, 'branch': branch, 'tracked_diff_sha256': diff.hexdigest(),
            'tracked_diff_bytes': size[0],
            'untracked_files': [{'path': name, 'sha256': sha256_file(repo / name, deadline)} for name in untracked],
            'clean': not size[0] and not untracked}


def read_pins(path: Path) -> dict[str, str]:
    """``sha256sum`` lines: digest, then the file name relative to the checkpoint."""
    pins = {}
    for number, line in enumerate(read_bounded(path).decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        pin = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if pin is None:
            raise HarnessError(f'{path}:{number}: expected "<sha256>  <file>"')
        pins[pin[2]] = pin[1].lower()
    return pins


def visible_files(directory: Path, deadline: Deadline) -> list[Path]:
    """Regular files under ``directory`` in sorted order. Hidden entries are pruned before descent, directory
    symlinks are not followed, and listing stops past MAX_LISTED_FILES visited entries."""
    found, pending, visited = [], [directory], 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                deadline.left()
                visited += 1
                if visited > MAX_LISTED_FILES:
                    raise HarnessError(f'{directory} holds more than {MAX_LISTED_FILES} entries')
                if entry.name.startswith('.'):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file():
                    found.append(Path(entry.path))
    return sorted(found)


def checkpoint_identity(directory: Path, pins_path: Path, deadline: Deadline) -> dict:
    """Every visible file by size and mtime; small files rehashed against their pins, large ones named unrehashed."""
    deadline.left()
    pins = read_pins(pins_path)
    files = []
    for path in visible_files(directory, deadline):
        deadline.left()
        relative = str(path.relative_to(directory))
        status = path.stat()
        entry = {'path': relative, 'bytes': status.st_size, 'mtime_ns': status.st_mtime_ns,
                 'pinned_sha256': pins.get(relative)}
        if status.st_size <= HASHED_FILE_LIMIT:
            entry['sha256'] = sha256_file(path, deadline)
        files.append(entry)
    present = {entry['path'] for entry in files}
    config_path = directory / 'config.json'
    try:
        deadline.left()
        config = json.loads(read_bounded(config_path)) if config_path.is_file() else {}
        declared = {'dtype': config.get('dtype', config.get('torch_dtype')),
                    'quantization_config': config.get('quantization_config')}
    except (ValueError, AttributeError) as error:
        raise HarnessError(f'{config_path} is not a JSON object: {error}') from None
    return {'path': str(directory), 'pins_path': str(pins_path), 'pins_sha256': sha256_file(pins_path, deadline),
            'files': files,
            'pin_mismatches': [e['path'] for e in files if 'sha256' in e and e['pinned_sha256'] not in (None, e['sha256'])],
            'pinned_but_missing': sorted(set(pins) - present),
            'not_rehashed': [e['path'] for e in files if 'sha256' not in e],
            'unpinned_not_rehashed': [e['path'] for e in files if 'sha256' not in e and e['pinned_sha256'] is None],
            'declared_precision': declared}


def pilot_identity(directory: Path, cmdline: list[str], deadline: Deadline) -> dict:
    """The recipe pilot's launch record: process start to /health readiness is startup, not a request."""
    deadline.left()
    try:
        ready = json.loads(read_bounded(directory / 'ready.json'))
        record = {'path': str(directory), 'backend': ready.get('backend'), 'unit': ready.get('unit'),
                  'server_command': ready.get('server_command'),
                  'server_command_equals_process_cmdline': ready.get('server_command') == cmdline,
                  'process_start_to_ready_s': ready.get('cold_ready_s'), 'ready_at_epoch': ready.get('ready_at_epoch')}
    except (OSError, ValueError, AttributeError) as error:
        raise HarnessError(f'no readable pilot ready.json object in {directory}: {error}') from None
    return record


def read_records(sources: list[Path], deadline: Deadline) -> list[tuple[Path, bytes]]:
    """Each file whole, together at most MAX_RECORD_BYTES."""
    records, budget = [], MAX_RECORD_BYTES
    for source in sources:
        deadline.left()
        data = read_bounded(source, budget)
        budget -= len(data)
        records.append((source, data))
    return records


def write_records(records: list[tuple[Path, bytes]], destination: Path) -> list[dict]:
    destination.mkdir()
    written = []
    for index, (source, data) in enumerate(records):
        target = destination / f'{index:02d}-{source.name}'
        target.write_bytes(data)
        written.append({'source': str(source), 'copy': str(target.relative_to(destination.parent)),
                        'sha256': sha256_bytes(data)})
    return written


# Statistics

def percentiles(ordered: list, total: int) -> dict:
    """Nearest-rank p50/p95/p99 over ``total`` attempts; ranks past the ``ordered`` successes are failures."""
    ranks = {f'p{p}': max(1, -(-p * total // 100)) for p in (50, 95, 99)}
    return {name: ordered[rank - 1] if rank <= len(ordered) else 'failure' for name, rank in ranks.items()}


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else None
    return {'n': len(ordered), 'mean': mean, 'stdev': stdev, 'cv': stdev / mean if stdev is not None and mean else None,
            'min': ordered[0], **percentiles(ordered, len(ordered)), 'max': ordered[-1]}


def censored(values: list[float], failed: int) -> dict:
    """Percentiles with every failure and timeout in the denominator, ranked slower than any success."""
    total = len(values) + failed
    return {'attempts': total, 'failed': failed, **percentiles(sorted(values), total)}


def summarize(samples: list[dict], scheduled: int, stream: bool) -> dict:
    phases = {}
    for phase in PHASES:
        rows = [s for s in samples if s['phase'] == phase]
        if not rows:
            continue
        ok = [s for s in rows if s['outcome'] == 'ok']
        failed = len(rows) - len(ok)
        audio_hashes: dict[str, set] = {}
        for s in ok:
            audio_hashes.setdefault(s['item'], set()).add(s['body_sha256'])
        durations = [s['audio']['duration_s'] for s in ok]
        phases[phase] = {
            'attempted': len(rows), 'succeeded': len(ok), 'success_rate': len(ok) / len(rows),
            'failures': dict(sorted(Counter(s['outcome'] for s in rows if s['outcome'] != 'ok').items())),
            'end_to_end_s': {'all_attempts': censored([s['total_s'] for s in ok], failed),
                             'ok_only': stats([s['total_s'] for s in ok])},
            'realtime_factor': {'all_attempts': censored([s['rtf'] for s in ok], failed),
                                'ok_only': stats([s['rtf'] for s in ok])},
            'audio_s': {'total': sum(durations), 'ok_only': stats(durations)},
            'clipped_samples': sum(s['audio']['clipped_samples'] for s in ok),
            'items_with_varying_audio': sorted(item for item, hashes in audio_hashes.items() if len(hashes) > 1),
        }
        if stream:
            phases[phase]['first_frame_s'] = {'all_attempts': censored([s['first_frame_s'] for s in ok], failed),
                                              'ok_only': stats([s['first_frame_s'] for s in ok])}
    playable = {'reported': True, 'definition': FIRST_PLAYABLE} if stream else {'reported': False, 'reason': NO_FIRST_PLAYABLE}
    return {'timing_scope': TIMING_SCOPE, 'realtime_factor_definition': 'end_to_end_s / audio_s',
            'first_playable_audio': playable,
            'scheduled': scheduled, 'completed': len(samples), 'complete': len(samples) == scheduled,
            'phases': phases}


# run

def run(args: argparse.Namespace) -> int:
    endpoint = Endpoint(args.endpoint)
    items, corpus_sha256 = load_corpus(args.corpus)
    schedule = plan(items, args.server_start == 'fresh', args.warmup, args.repeats)
    contract, limit = contract_of(args)
    deadline = Deadline(IDENTITY_DEADLINE_S, f'identity collection ({IDENTITY_DEADLINE_S} s)')
    pilot_dir = args.pilot_evidence.resolve() if args.pilot_evidence else None
    try:
        process = process_identity(args.server_pid, deadline)
        checkpoint = checkpoint_identity(args.checkpoint.resolve(), args.checkpoint_pins.resolve(), deadline)
        source = source_identity(args.server_source.resolve(), deadline)
        pilot = pilot_identity(pilot_dir, process['cmdline'], deadline) if pilot_dir else None
        configs = read_records(args.config, deadline)
        pilot_records = read_records([p for p in (pilot_dir / 'session.json', pilot_dir / 'ready.json')
                                      if p.is_file()] if pilot_dir else [], deadline)
    except TimeoutError as error:
        raise HarnessError(str(error)) from None
    if checkpoint['pin_mismatches'] or checkpoint['pinned_but_missing']:
        raise HarnessError(f'checkpoint differs from its pins: mismatched {checkpoint["pin_mismatches"]}, '
                           f'missing {checkpoint["pinned_but_missing"]}')
    checkpoint['path_in_server_cmdline'] = checkpoint['path'] in process['cmdline']
    before = server_state(endpoint)
    if before['health'].get('status') != 200:
        raise HarnessError(f'GET /health is not 200 ({before["health"]}); no speech request was sent')

    evidence = args.evidence_dir
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / 'bodies').mkdir()
    manifest = {
        'label': args.label, 'harness_sha256': sha256_bytes(read_bounded(Path(__file__).resolve())),
        'argv': sys.argv, 'python': sys.version, 'platform': platform.platform(), 'started_at_epoch': time.time(),
        'endpoint': endpoint.url, 'model': args.model, 'contract': contract, 'timeout_s': args.timeout,
        'response_byte_limit': limit,
        'worst_case_network_wall_s': {'identity': IDENTITY_DEADLINE_S, 'snapshots': 6 * SNAPSHOT_TIMEOUT_S,
                                      'requests': len(schedule) * args.timeout},
        'timing_scope': TIMING_SCOPE, 'client_concurrency': 1, 'connection': 'new connection per request',
        'stated_precision': args.precision, 'server_start': args.server_start,
        'corpus': {'path': str(args.corpus.resolve()), 'items': len(items)},
        'corpus_sha256': corpus_sha256, 'schedule': [[phase, r, item['id']] for phase, r, item in schedule],
        'identity': {'process': process, 'source': source, 'checkpoint': checkpoint, 'pilot': pilot,
                     'configs': write_records(configs, evidence / 'configs'),
                     'pilot_records': write_records(pilot_records, evidence / 'pilot')},
    }
    write_json(evidence / 'manifest.json', manifest)
    write_json(evidence / 'server-before.json', before)

    samples: list[dict] = []  # at most MAX_REQUESTS
    try:
        with open(evidence / 'samples.jsonl', 'w') as log:
            for seq, (phase, round_index, item) in enumerate(schedule, 1):
                sample = measure_request(endpoint, args.model, contract, args.timeout, limit, evidence,
                                         seq, phase, round_index, item)
                log.write(json.dumps(sample) + '\n')
                log.flush()
                samples.append(sample)
                audio = f'{sample["audio"]["duration_s"]:.3f}s audio' if sample['audio'] else sample['detail'] or ''
                print(f'{seq:>4}/{len(schedule)} {phase:<6} r{round_index} {item["id"]:<10} '
                      f'{sample["outcome"]:<15} {sample["total_s"]:8.3f}s {audio}'[:200], flush=True)
    finally:
        try:
            unchanged = process_start_ticks(args.server_pid) == process['start_ticks']
        except OSError:
            unchanged = False
        write_json(evidence / 'server-after.json', {**server_state(endpoint), 'server_process_unchanged': unchanged})
        summary = summarize(samples, len(schedule), contract['stream'])
        summary.update(label=args.label, server_process_unchanged=unchanged)
        write_json(evidence / 'summary.json', summary)
    print(json.dumps({phase: {k: v[k] for k in ('attempted', 'succeeded', 'failures')}
                      for phase, v in summary['phases'].items()}), flush=True)
    clean = summary['complete'] and unchanged and all(s['outcome'] == 'ok' for s in samples)
    return 0 if clean else 1


def measure_request(endpoint: Endpoint, model: str, contract: dict, timeout: float, limit: int, evidence: Path,
                    seq: int, phase: str, round_index: int, item: dict) -> dict:
    text = item['text']
    request = {'model': model, 'input': text, 'voice': contract['voice'], 'response_format': contract['response_format']}
    if contract['stream']:
        request['stream'] = True
    sample = {'seq': seq, 'phase': phase, 'round': round_index, 'item': item['id'],
              'input_sha256': sha256_bytes(text.encode()), 'input_chars': len(text), 'started_at_epoch': time.time()}
    result, data = exchange(endpoint, 'POST', SPEECH_PATH, json.dumps(request).encode(), timeout, limit,
                            2 * contract['frame_samples'] if contract['stream'] else None)
    sample.update(result, body_sha256=None, body_file=None, audio=None, rtf=None)
    if data is None:
        return sample
    sample['body_sha256'] = sha256_bytes(data)
    if sample['status'] != 200:
        sample.update(outcome=f'http_{sample["status"]}', detail=error_message(data))
        return sample
    name = f'bodies/{seq:04d}-{item["id"]}.{contract["response_format"]}'
    (evidence / name).write_bytes(data)
    sample['body_file'] = name
    try:
        sample['audio'] = audio_of(data, sample['content_type'], contract)
    except AudioError as error:
        sample.update(outcome='invalid_audio', detail=str(error))
        return sample
    sample['rtf'] = sample['total_s'] / sample['audio']['duration_s']
    return sample


# compare

def load_run(directory: Path) -> dict:
    try:
        manifest_bytes = read_bounded(directory / 'manifest.json')
        manifest = json.loads(manifest_bytes)
        summary = json.loads(read_bounded(directory / 'summary.json'))
        samples = [json.loads(line) for line in read_bounded(directory / 'samples.jsonl').splitlines() if line]
    except (OSError, ValueError) as error:
        raise HarnessError(f'{directory} is not a readable run: {error}') from None
    if not summary['complete'] or [[s['phase'], s['round'], s['item']] for s in samples] != manifest['schedule']:
        raise HarnessError(f'{directory} is incomplete or its samples do not follow its schedule')
    return {'dir': directory, 'manifest': manifest, 'manifest_sha256': sha256_bytes(manifest_bytes),
            'summary': summary, 'samples': {(s['phase'], s['round'], s['item']): s for s in samples}}


def compare(args: argparse.Namespace) -> int:
    a, b = load_run(args.a), load_run(args.b)
    for key in ('corpus_sha256', 'schedule', 'contract', 'timeout_s'):
        if a['manifest'][key] != b['manifest'][key]:
            raise HarnessError(f'runs differ in {key}; their request cohorts are not matched')
    phases = {}
    for phase in PHASES:
        keys = [key for key in a['samples'] if key[0] == phase]
        if not keys:
            continue
        pairs = [(k, a['samples'][k], b['samples'][k]) for k in keys]
        both = [(x, y) for _, x, y in pairs if x['outcome'] == y['outcome'] == 'ok']
        unmatched = [{'round': k[1], 'item': k[2], 'a': x['outcome'], 'b': y['outcome']}
                     for k, x, y in pairs if not x['outcome'] == y['outcome'] == 'ok']
        outcomes = Counter((u['a'] == 'ok', u['b'] == 'ok') for u in unmatched)
        phases[phase] = {
            'pairs': len(keys), 'both_ok': len(both), 'a_only_ok': outcomes[(True, False)],
            'b_only_ok': outcomes[(False, True)], 'both_failed': outcomes[(False, False)],
            'unmatched': unmatched,
            'all_attempts_end_to_end_s': {name: run['summary']['phases'][phase]['end_to_end_s']['all_attempts']
                                          for name, run in (('a', a), ('b', b))},
            'matched_end_to_end_s': {'a': stats([x['total_s'] for x, _ in both]),
                                     'b': stats([y['total_s'] for _, y in both]),
                                     'b_over_a': stats([y['total_s'] / x['total_s'] for x, y in both])},
            'matched_realtime_factor': {'a': stats([x['rtf'] for x, _ in both]), 'b': stats([y['rtf'] for _, y in both])},
            'matched_identical_audio': sum(x['body_sha256'] == y['body_sha256'] for x, y in both),
        }

    def side(run: dict) -> dict:
        identity = run['manifest']['identity']
        return {'evidence': str(run['dir']), 'label': run['manifest']['label'], 'manifest_sha256': run['manifest_sha256'],
                'harness_sha256': run['manifest']['harness_sha256'], 'server_exe_sha256': identity['process']['exe_sha256'],
                'source_head': identity['source']['head'], 'source_diff_sha256': identity['source']['tracked_diff_sha256'],
                'checkpoint_pins_sha256': identity['checkpoint']['pins_sha256'],
                'stated_precision': run['manifest']['stated_precision'],
                'server_process_unchanged': run['summary']['server_process_unchanged']}

    report = {'timing_scope': TIMING_SCOPE, 'a': side(a), 'b': side(b), 'phases': phases}
    with open(args.out, 'x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({p: {k: v[k] for k in ('pairs', 'both_ok', 'a_only_ok', 'b_only_ok', 'both_failed')}
                      for p, v in phases.items()}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    r = commands.add_parser('run', help='measure one server process over a corpus schedule')
    r.add_argument('--endpoint', required=True, help='base URL, e.g. http://IP:PORT (an IP literal: no DNS)')
    r.add_argument('--model', required=True, help='the served model id sent in every request')
    r.add_argument('--corpus', type=Path, required=True, help='JSONL of {"id", "text"} in send order')
    r.add_argument('--evidence-dir', type=Path, required=True, help='new directory for this run')
    r.add_argument('--label', required=True, help='what this run is, e.g. its evidence class')
    r.add_argument('--precision', required=True, help='the stated serving precision')
    r.add_argument('--server-pid', type=int, required=True, help='the serving process, inspected through /proc')
    r.add_argument('--server-source', type=Path, required=True, help='git checkout the server was built from')
    r.add_argument('--checkpoint', type=Path, required=True)
    r.add_argument('--checkpoint-pins', type=Path, required=True, help='sha256sum-format pins for checkpoint files')
    r.add_argument('--pilot-evidence', type=Path, help="the recipe pilot's evidence directory (ready.json)")
    r.add_argument('--config', type=Path, action='append', default=[], help='server config file to record; repeatable')
    r.add_argument('--server-start', choices=('fresh', 'reused'), required=True,
                   help='fresh: no speech request has reached this process, so the first request is cold')
    r.add_argument('--warmup', type=int, default=2, help='unscored requests after the cold one (default 2)')
    r.add_argument('--repeats', type=int, default=3, help='scored passes over the corpus (default 3)')
    r.add_argument('--timeout', type=float, default=120.0, help='whole-request deadline in seconds (default 120)')
    r.add_argument('--response-format', choices=tuple(MEDIA_TYPES), default='wav')
    r.add_argument('--stream', action='store_true',
                   help='send "stream": true (needs --response-format pcm); a 200 must be a chunked PCM stream')
    r.add_argument('--sample-rate', type=int, default=44100, help='contracted rate (Drama 3: 44100)')
    r.add_argument('--frame-samples', type=int, default=2048, help='samples per codec frame (Drama 3: 2048)')
    r.add_argument('--max-frames', type=int, default=1024, help='frame limit per request (Drama 3: 1024)')
    c = commands.add_parser('compare', help='match two complete runs request by request')
    c.add_argument('a', type=Path)
    c.add_argument('b', type=Path)
    c.add_argument('--out', type=Path, required=True, help='new JSON file')
    args = parser.parse_args()
    try:
        return run(args) if args.command == 'run' else compare(args)
    except HarnessError as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2
    except (OSError, UnicodeDecodeError) as error:  # unreadable inputs, or an evidence path that already exists
        print(f'error: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
