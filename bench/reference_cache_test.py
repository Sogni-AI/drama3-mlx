#!/usr/bin/env python3
"""Deterministic check for the reference-prefill cache: a voice's first request builds the prefix snapshot and the
next request with the same voice restores it, and both replies are byte-identical.

Usage: reference_cache_test.py --base http://HOST:PORT --log PATH [--voice NAME] [--input TEXT]

Point it at a server started by recipe/pilot.py or recipe/pilot_macos.py; PATH is that server's log. The check reads
the log only for the cache lines the patched host writes ("reference cache built", "reference cache hit"), so it fails
on an engine without the cache. Stdlib only.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_INPUT = "The train leaves at seven fifteen, and this is the voice you came for."


def synth(base: str, voice: str, text: str) -> bytes:
    body = json.dumps({"model": "fishaudio/drama-3", "input": text, "voice": voice,
                       "response_format": "pcm"}).encode()
    request = urllib.request.Request(base.rstrip("/") + "/v1/audio/speech", body,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            if response.status != 200:
                raise SystemExit(f"reference_cache_test: HTTP {response.status}")
            return response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"reference_cache_test: HTTP {error.code}: {error.read()[:200]!r}")


def lines(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"reference_cache_test: no log at {path}")
    return path.read_text(errors="replace").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="http://HOST:PORT of the running server")
    parser.add_argument("--log", required=True, type=Path, help="that server's log file")
    parser.add_argument("--voice", default="jones", help="a catalog voice, not 'default'")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    args = parser.parse_args()

    before = len(lines(args.log))
    first = synth(args.base, args.voice, args.input)
    built = [line for line in lines(args.log)[before:] if "reference cache built" in line]
    if not built:
        raise SystemExit("reference_cache_test: FAIL - the first request did not build a prefix snapshot")

    mark = len(lines(args.log))
    second = synth(args.base, args.voice, args.input)
    after = lines(args.log)[mark:]
    hit = [line for line in after if "reference cache hit" in line]
    if not hit:
        raise SystemExit("reference_cache_test: FAIL - the second request did not restore the snapshot")
    if second != first:
        raise SystemExit("reference_cache_test: FAIL - the cached reply differs from the uncached one")

    print(json.dumps({"built": built[-1].strip(), "hit": hit[-1].strip(),
                      "identical": True, "bytes": len(first), "audio_seconds": len(first) / 2 / 44100}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
