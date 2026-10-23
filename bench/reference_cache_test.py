#!/usr/bin/env python3
"""Check reference-voice PCM identity across default, alternate-voice and text transitions.

Start a single-lane server with the builtin default voice and two synthetic or consented
catalog voices known to produce distinct PCM. This is a real-model regression, not a speed benchmark:

  reference_cache_test.py --base http://HOST:PORT --voice A --other-voice B --out result.json

It checks replies, not implementation-specific log wording. Use corpus_capture.py with
--voice on the baseline and candidate for the separate cross-build identity gate.
"""
import argparse
import hashlib
import json
import socket
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
                raise RuntimeError(f"HTTP {response.status}")
            body = response.read()
            if response.headers.get("X-Finish-Reason") != "stop":
                raise RuntimeError("speech did not finish normally")
            if response.headers.get_content_type() != "audio/pcm" or not body or len(body) % 2:
                raise RuntimeError("invalid PCM response")
            return body
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read()[:200]!r}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="http://HOST:PORT of the running server")
    parser.add_argument("--voice", required=True, help="first reference catalog voice")
    parser.add_argument("--other-voice", required=True, help="different reference catalog voice")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", required=True, type=Path, help="new report path")
    args = parser.parse_args()
    if args.voice.lower() == "default" or args.other_voice.lower() == "default" or args.voice == args.other_voice:
        parser.error("provide two distinct catalog names, neither default")
    if args.out.exists():
        parser.error("--out already exists")
    report = {"voices": [args.voice, args.other_voice], "requests": [], "checks": {}, "errors": []}

    def capture(label, voice, text=args.input):
        pcm = synth(args.base, voice, text)
        report["requests"].append({"label": label, "voice": voice, "input": text,
                                   "bytes": len(pcm), "sha256": hashlib.sha256(pcm).hexdigest()})
        return pcm

    try:
        first = capture("first", args.voice)
        report["checks"]["repeat"] = capture("repeat", args.voice) == first
        builtin = capture("builtin-default", "default")
        report["checks"]["builtin_is_distinct"] = builtin != first
        report["checks"]["after_default"] = capture("after-default", args.voice) == first
        other = capture("other-first", args.other_voice)
        report["checks"]["other_voice_is_distinct"] = other != first
        report["checks"]["after_other"] = capture("after-other", args.voice) == first
        different_text = "The library opens at nine. Please bring the blue notebook tomorrow."
        if different_text == args.input:
            different_text = DEFAULT_INPUT
        varied = capture("varied-text", args.voice, different_text)
        report["checks"]["varied_text_is_distinct"] = varied != first
        report["checks"]["after_varied_text"] = capture("after-varied-text", args.voice) == first
        report["checks"]["varied_text_repeat"] = capture("varied-text-repeat", args.voice, different_text) == varied
        for cycle in range(4):
            report["checks"][f"other_{cycle}"] = capture(f"other-{cycle}", args.other_voice) == other
            report["checks"][f"first_{cycle}"] = capture(f"first-{cycle}", args.voice) == first
    except (RuntimeError, urllib.error.URLError, socket.timeout) as error:
        report["errors"].append(str(error))
    report["passed"] = not report["errors"] and all(report["checks"].values())
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
