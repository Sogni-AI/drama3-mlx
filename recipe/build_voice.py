#!/usr/bin/env python3
"""Build a Drama 3 HTTP voice catalog file from encoded reference clips.

Usage:
  drama3-generate --codec drama-3/codec.pth --encode out.codes --ref-audio clip.wav
  python3 voices/build_voice.py NAME NOTE TRANSCRIPT:CODES [TRANSCRIPT:CODES ...]

Writes runtime/voices/NAME.json in the drama3-http-voice/1 format the patched server
loads with --voices. Transcripts must match their clips; they are the conditioning text.
"""
import json
import struct
import sys
from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent / 'runtime/voices'


def read_codes(path: Path) -> list[list[int]]:
    raw = path.read_bytes()
    if len(raw) % 40:
        sys.exit(f'{path}: size {len(raw)} is not a multiple of 40 bytes (10 int32 rows)')
    rows = [list(struct.unpack('<10i', raw[i:i + 40])) for i in range(0, len(raw), 40)]
    for row in rows:
        if not 0 <= row[0] < 4096 or any(not 0 <= v < 1024 for v in row[1:]):
            sys.exit(f'{path}: code out of range {row}')
    return rows


def main() -> int:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    name, note, *pairs = sys.argv[1:]
    if not name.replace('-', '').replace('_', '').isalnum() or name.lower() == 'default':
        sys.exit('name must be alphanumeric/-/_ and not "default"')
    references = []
    for pair in pairs:
        transcript, _, codes = pair.rpartition(':')
        if not transcript or not codes:
            sys.exit(f'bad pair {pair!r}: expected TRANSCRIPT:CODES')
        rows = read_codes(Path(codes))
        references.append({'text': transcript, 'codes': rows})
        print(f'{Path(codes).name}: {len(rows)} frames ({len(rows)/21.53:.1f}s)')
    doc = {
        'format': 'drama3-http-voice/1',
        'source': {'kind': 'consented', 'note': note},
        'references': references,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f'{name}.json'
    target.write_text(json.dumps(doc, separators=(',', ':')) + '\n')
    print(f'{target}: {target.stat().st_size} bytes, {len(references)} reference(s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
