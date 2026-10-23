#!/usr/bin/env python3
"""HTTP-level check of streamed speech against the whole reply of the SAME running server (stdlib only).

Run it against a native server that is already up, for example one started by recipe/pilot.py; this script starts no
server, model or GPU work. For one input it sends: whole WAV, whole PCM, streamed PCM (raw socket, own HTTP/1.1
chunk parser, arrival time of every received block), then a streamed request it abandons once the first frame has
arrived, then the whole PCM again. Streaming passes only if the streamed body is byte-identical to the whole PCM body
and to the WAV body after its 44-byte header, every chunk is whole frames, the terminating chunk arrived, the
abandoned stream is incomplete, its slot frees, and the repeat whole reply is unchanged. Timing is measured per
audio byte from the received blocks, never from HTTP chunk boundaries, and is never a pass condition.
--self-test exercises the parser and the timing on byte-string fixtures only."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

FRAME_BYTES = 2048 * 2


def parse_chunked(data):
    """Chunks of an HTTP/1.1 chunked body: (chunks, complete). complete only when the zero chunk and its CRLF end it."""
    chunks, at = [], 0
    while True:
        line_end = data.find(b'\r\n', at)
        if line_end < 0:
            return chunks, False
        size = int(data[at:line_end].split(b';')[0], 16)
        start = line_end + 2
        if size == 0:
            return chunks, data[start:start + 2] == b'\r\n' and start + 2 == len(data)
        if len(data) < start + size + 2:
            return chunks, False
        if data[start + size:start + size + 2] != b'\r\n':
            raise ValueError('chunk not followed by CRLF')
        chunks.append((at, data[start:start + size]))
        at = start + size + 2


def audio_available(data):
    """Audio bytes received so far in a chunked body, counting the received part of an unfinished chunk; HTTP
    completeness stays parse_chunked's (terminator only)."""
    count, at = 0, 0
    while True:
        line_end = data.find(b'\r\n', at)
        if line_end < 0:
            return count
        size = int(data[at:line_end].split(b';')[0], 16)
        start = line_end + 2
        if size == 0 or len(data) < start + size + 2:
            return count + min(size, max(len(data) - start, 0))
        count += size
        at = start + size + 2


def abandon_now(raw):
    """The live abandonment decision: frame 0's bytes are all here, whatever chunk they arrive in."""
    split = raw.find(b'\r\n\r\n')
    return split >= 0 and b'transfer-encoding: chunked' in raw[:split].lower() and audio_available(raw[split + 4:]) >= FRAME_BYTES


def whole(base, body, busy_retry_s=0.0):
    """One whole reply; a 503 (the engine still busy) is retried every 0.2 s for `busy_retry_s`."""
    deadline, retries = time.monotonic() + busy_retry_s, 0
    while True:
        request = urllib.request.Request(base + '/v1/audio/speech', data=json.dumps(body).encode(), method='POST',
                                         headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=600) as reply:
                return {'status': reply.status, 'content_type': reply.headers['Content-Type'], 'bytes': reply.read(),
                        'seconds': time.monotonic() - started, 'busy_retries': retries}
        except urllib.error.HTTPError as error:
            if error.code != 503 or time.monotonic() >= deadline:
                raise
            retries += 1
            time.sleep(0.2)


def split_reply(raw):
    """Status line, headers, head length, chunks and completeness of a (possibly partial) reply."""
    split = raw.find(b'\r\n\r\n')
    if split < 0:
        return None
    lines = raw[:split].decode('latin-1').split('\r\n')
    headers = {k.lower(): v.strip() for k, v in (line.split(':', 1) for line in lines[1:])}
    chunked = headers.get('transfer-encoding') == 'chunked'
    chunks, complete = parse_chunked(raw[split + 4:]) if chunked else ([], False)
    return lines[0], headers, split + 4, chunks, complete


def payload_counts(arrivals, head_len, chunks):
    """(audio bytes fully received, seconds) after each received block, whatever the chunk boundaries."""
    segments = [(head_len + at + len(f'{len(c):x}') + 2, len(c)) for at, c in chunks]
    return [(sum(min(max(n - start, 0), size) for start, size in segments), t) for n, t in arrivals]


def moments(arrivals, head_len, chunks, complete, rate, frame_bytes=FRAME_BYTES):
    """Client-side moments, each from the arrival of the audio bytes themselves:
    first byte of the reply; first whole sample; first whole frame (first decodable audio); the earliest playback
    start from which no sample arrives after the playhead needs it (gap-free playable, sample-exact); completion
    (terminating chunk). chunk_buffered_playback_start_s is the conservative start of a player that only consumes
    whole HTTP chunks; it depends on chunk boundaries and is reported only beside the byte-level start."""
    if not chunks:
        return None
    counts = payload_counts(arrivals, head_len, chunks)
    first = lambda need: next(t for c, t in counts if c >= need)
    gapless, before = float('-inf'), 0
    for count, t in counts:
        if count // 2 > before // 2:  # samples before//2 .. count//2 - 1 completed in this block
            gapless = max(gapless, t - (before // 2) / rate)
        before = count
    buffered, offset = float('-inf'), 0
    for at, c in chunks:
        end = head_len + at + len(f'{len(c):x}') + 2 + len(c)
        buffered = max(buffered, next(t for n, t in arrivals if n >= end) - offset / 2 / rate)
        offset += len(c)
    return {'first_byte_s': arrivals[0][1], 'first_sample_s': first(2), 'first_frame_s': first(frame_bytes),
            'earliest_gapless_playback_start_s': gapless, 'chunk_buffered_playback_start_s': buffered,
            'completion_s': arrivals[-1][1] if complete else None, 'audio_s': counts[-1][0] / 2 / rate}


def streamed(base, body, abandon_after_first_frame=False):
    url = urllib.parse.urlsplit(base)
    payload = json.dumps(body).encode()
    sock = socket.create_connection((url.hostname, url.port), timeout=600)
    head = (f'POST /v1/audio/speech HTTP/1.1\r\nHost: {url.netloc}\r\nContent-Type: application/json\r\n'
            f'Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n').encode()
    started = time.monotonic()
    sock.sendall(head + payload)
    raw, arrivals = b'', []  # (bytes received so far, seconds since the request was sent)
    while True:
        got = sock.recv(1 << 16)
        if not got:
            break
        raw += got
        arrivals.append((len(raw), time.monotonic() - started))
        if abandon_after_first_frame and abandon_now(raw):
            break
    sock.close()
    return finalize(raw, arrivals, abandoned=abandon_after_first_frame)


def finalize(raw, arrivals, abandoned):
    """A received reply's report. An abandoned stream is cut on purpose, so its timing is not measured."""
    status, headers, head_len, chunks, complete = split_reply(raw)
    return {'status_line': status, 'headers': headers, 'chunk_bytes': [len(c) for _, c in chunks],
            'complete': complete, 'pcm': b''.join(c for _, c in chunks), 'audio_received': audio_available(raw[head_len:]),
            'moments': None if abandoned else moments(arrivals, head_len, chunks, complete, int(headers.get('x-sample-rate', 44100)))}


def fixture(chunk_sizes, cuts, rate=8):
    """A raw chunked reply carrying 0..n-1 audio bytes in `chunk_sizes`, received in blocks that end right after the
    audio byte counts in `cuts` (count, seconds); framing after a cut arrives with the next block."""
    head = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nX-Sample-Rate: %d\r\n\r\n' % rate
    body, positions, payload = b'', [], 0
    for size in chunk_sizes:
        body += f'{size:x}\r\n'.encode()
        positions += [len(head) + len(body) + i + 1 for i in range(size)]
        body += bytes(range(payload, payload + size)) + b'\r\n'
        payload += size
    raw = head + body + b'0\r\n\r\n'
    arrivals = [(positions[count - 1], t) for count, t in cuts[:-1]] + [(len(raw), cuts[-1][1])]
    _, headers, head_len, chunks, complete = split_reply(raw)
    return moments(arrivals, head_len, chunks, complete, rate, frame_bytes=4)


def self_test():
    body = b'4\r\nabcd\r\n2\r\nef\r\n0\r\n\r\n'
    assert parse_chunked(body) == ([(0, b'abcd'), (9, b'ef')], True)
    assert parse_chunked(body[:-2]) == ([(0, b'abcd'), (9, b'ef')], False)
    assert parse_chunked(b'4\r\nabcd\r\n') == ([(0, b'abcd')], False)
    assert parse_chunked(b'4\r\nab') == ([], False)
    # live abandonment: a coalesced 3-frame chunk of which frame 0 and a few more bytes have arrived, with neither
    # the rest of its audio nor its CRLF: abandon now, while the reply is still incomplete
    head = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n'
    partial = head + b'3000\r\n' + bytes(FRAME_BYTES + 10)
    assert abandon_now(partial) and parse_chunked(partial[len(head):]) == ([], False)
    assert not abandon_now(head + b'3000\r\n' + bytes(FRAME_BYTES - 1))
    split_frame = head + b'800\r\n' + bytes(2048) + b'\r\n3000\r\n' + bytes(2048)  # frame 0 across two chunks
    assert abandon_now(split_frame)
    # through the report the live path builds: one completed chunk, frame 0 received, incomplete, no timing
    cut = finalize(split_frame, [(len(head) + 10, 0.5), (len(split_frame), 1.0)], abandoned=True)
    assert cut['chunk_bytes'] == [2048] and cut['audio_received'] == FRAME_BYTES
    assert not cut['complete'] and cut['moments'] is None
    assert not abandon_now(b'HTTP/1.1 500 Internal Server Error\r\nContent-Length: 9000\r\n\r\n' + bytes(9000))
    # rate 8 samples/s, 4-byte frames: 12 audio bytes = 6 samples = 0.75 s of audio
    # one 12-byte chunk whose first frame arrives at 1 s and the rest at 3 s
    one = fixture([12], [(4, 1.0), (12, 3.0)])
    assert one['first_frame_s'] == 1.0 and one['first_sample_s'] == 1.0
    assert one['earliest_gapless_playback_start_s'] == 3.0 - 2 / 8  # sample 2 is needed 0.25 s after the start
    assert one['chunk_buffered_playback_start_s'] == 3.0  # the chunk ends with the last block
    # the same audio-byte timing under other chunk boundaries gives the same byte-level moments
    for sizes in ([4, 8], [4, 4, 4], [2, 10], [12]):
        other = fixture(sizes, [(4, 1.0), (12, 3.0)])
        assert {k: other[k] for k in ('first_sample_s', 'first_frame_s', 'earliest_gapless_playback_start_s', 'completion_s')} == \
               {k: one[k] for k in ('first_sample_s', 'first_frame_s', 'earliest_gapless_playback_start_s', 'completion_s')}, sizes
    assert fixture([4, 8], [(4, 1.0), (12, 3.0)])['chunk_buffered_playback_start_s'] == 3.0 - 4 / 2 / 8
    print(json.dumps({'self_test': 'passed', 'scope': 'chunk parser, live abandonment decision and byte-level timing on byte-string fixtures only'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', help='http://HOST:PORT of the running server')
    parser.add_argument('--model')
    parser.add_argument('--input')
    parser.add_argument('--voice', help='Explicit catalog voice, including cached-reference abandonment')
    parser.add_argument('--out', type=Path, help='new JSON report path (must not exist)')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (args.base and args.model and args.input and args.out):
        parser.error('--base, --model, --input and --out are required')
    base = {'model': args.model, 'input': args.input}
    if args.voice is not None:
        base['voice'] = args.voice
    wav = whole(args.base, {**base, 'response_format': 'wav'})
    pcm = whole(args.base, {**base, 'response_format': 'pcm'})
    stream = streamed(args.base, {**base, 'response_format': 'pcm', 'stream': True})
    left = streamed(args.base, {**base, 'response_format': 'pcm', 'stream': True}, abandon_after_first_frame=True)
    # the abandoned synthesis is cancelled at its next frame or codec boundary; until then the one slot is busy
    again = whole(args.base, {**base, 'response_format': 'pcm'}, busy_retry_s=60)
    rate = int(stream['headers'].get('x-sample-rate', 0))
    checks = {
        'whole_wav_200': wav['status'] == 200 and wav['content_type'] == 'audio/wav',
        'whole_pcm_equals_wav_data': pcm['bytes'] == wav['bytes'][44:],
        'stream_head': stream['status_line'].startswith('HTTP/1.1 200') and stream['headers'].get('content-type') == 'audio/pcm'
                       and stream['headers'].get('transfer-encoding') == 'chunked' and rate == 44100
                       and stream['headers'].get('x-channels') == '1' and stream['headers'].get('x-bit-depth') == '16'
                       and 'content-length' not in stream['headers'],
        'stream_complete': stream['complete'],
        'stream_whole_frames': all(n % FRAME_BYTES == 0 for n in stream['chunk_bytes']),
        'stream_equals_whole_pcm': stream['pcm'] == pcm['bytes'],
        'abandoned_stream_incomplete': not left['complete'] and left['audio_received'] >= FRAME_BYTES,
        'slot_free_after_abandon': again['status'] == 200,
        'repeat_whole_unchanged': again['bytes'] == pcm['bytes'],
    }
    report = {'input': args.input, 'voice': args.voice, 'checks': checks, 'passed': all(checks.values()),
              'samples': len(pcm['bytes']) // 2, 'pcm_sha256': hashlib.sha256(pcm['bytes']).hexdigest(),
              'stream_chunk_bytes': stream['chunk_bytes'], 'moments': stream['moments'],
              'whole_pcm_seconds': pcm['seconds'], 'abandoned_audio_received_bytes': left['audio_received'],
              'busy_retries_after_abandon': again['busy_retries']}
    with open(args.out, 'x') as f:
        f.write(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'passed': report['passed'], **checks}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
