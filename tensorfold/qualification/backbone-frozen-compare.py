#!/usr/bin/env python3
"""CPU-only comparison of a new native oracle run with the unchanged frozen SGLang baseline."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import torch

HELPER_PATH = Path(__file__).resolve().with_name("backbone-reference-check.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_run", type=Path, help="directory whose native/ holds the backbone oracle's output")
    parser.add_argument("guard", type=Path, help="run-bounded-oracle.py evidence directory of that run")
    parser.add_argument("--baseline", type=Path, required=True, help="frozen backbone-reference-check.py output")
    parser.add_argument("--engine", type=Path, required=True, help="patched TensorFold tree the oracle was built in")
    args = parser.parse_args()
    output = args.native_run / "comparison.json"
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    spec = importlib.util.spec_from_file_location("frozen_backbone_comparison", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = {"schema": "drama-backbone-frozen-reference/1", "evidence_valid": False, "passed": False,
              "errors": [], "thresholds": helper.THRESHOLDS, "limits": helper.LIMITS,
              "reference_reexecuted": False, "script_sha256": helper.sha256(Path(__file__)),
              "comparison_helper_sha256": helper.sha256(HELPER_PATH)}
    try:
        baseline = json.loads((args.baseline / "results.json").read_text())
        fixture = json.loads((args.baseline / "fixture.json").read_text())
        if not baseline["evidence_valid"] or helper.THRESHOLDS != baseline["thresholds"]:
            raise ValueError("invalid frozen baseline or changed qualification thresholds")
        if helper.sha256(args.baseline / "fixture.json") != baseline["fixture_sha256"]:
            raise ValueError("frozen fixture hash changed")
        if helper.sha256(args.baseline / "tokens.u32le") != baseline["tokens_sha256"]:
            raise ValueError("frozen token hash changed")
        for name, expected in baseline["reference"]["files"].items():
            if helper.sha256(args.baseline / name) != expected:
                raise ValueError(f"frozen reference artifact changed: {name}")
        metadata_path = args.native_run / "native/native-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        expected = {"rows": fixture["tokens"], "hidden_width": helper.HIDDEN,
                    "eos": fixture["eos"], "semantic_start": fixture["semantic_start"],
                    "semantic_end": fixture["semantic_end"]}
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("native dimensions or semantic range disagree with fixture")
        completion = json.loads((args.guard / "completion.json").read_text())
        if completion["returncode"] != 0 or completion["failures"] or completion["stopped_unit_state"]["ActiveState"] != "inactive":
            raise ValueError("native guard did not complete cleanly")
        native_files = {}
        for chunk in helper.CHUNKS:
            for kind, size in (("hidden.bf16", fixture["tokens"] * helper.HIDDEN * 2),
                               ("semantic.f32", helper.CANDIDATES * 4)):
                name = f"native-chunk-{chunk}-{kind}"
                path = args.native_run / "native" / name
                if path.stat().st_size != size:
                    raise ValueError(f"native artifact size mismatch: {name}")
                native_files[name] = helper.sha256(path)
        independent_chunk_control = all(len({native_files[f"native-chunk-{c}-{kind}"] for c in helper.CHUNKS}) == 1
                                        for kind in ("hidden.bf16", "semantic.f32"))
        if not (metadata["chunk_bit_exact"] and metadata["reset_bit_exact"] and independent_chunk_control):
            raise ValueError("native chunk/reset controls failed")
        reference = {"hidden_bits": helper.bf16_bits(args.baseline / "reference-hidden.bf16", (fixture["tokens"], helper.HIDDEN)),
                     "semantic": np.fromfile(args.baseline / "reference-semantic.f32", dtype="<f4")}
        with contextlib.redirect_stdout(io.StringIO()):
            comparisons = helper.compare(args.native_run, fixture, reference)
        result.update(evidence_valid=True, passed=all(c["pass"] for c in comparisons.values()),
                      comparisons=comparisons, native_metadata=metadata, native_files=native_files,
                      independent_chunk_control=independent_chunk_control,
                      frozen_results_sha256=helper.sha256(args.baseline / "results.json"),
                      fixture_sha256=baseline["fixture_sha256"], tokens_sha256=baseline["tokens_sha256"],
                      reference_files=baseline["reference"]["files"],
                      guard_completion_sha256=helper.sha256(args.guard / "completion.json"),
                      current_oracle_binary_sha256=helper.sha256(args.engine / "zig-out/oracle/bin/tf-drama-backbone-oracle"),
                      current_serving_binary_sha256=helper.sha256(args.engine / "zig-out/native/bin/tensorfold-native"),
                      current_backbone_cuda_sha256=helper.sha256(args.engine / "zig/kernels/cuda/drama_backbone.cu"))
        result["cuda_initialized"] = torch.cuda.is_initialized()
        if result["cuda_initialized"]:
            raise ValueError("CPU comparison unexpectedly initialized CUDA")
    except Exception as exc:
        result.update(evidence_valid=False, passed=False)
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("evidence_valid", "passed", "errors")} |
                     {"results_sha256": helper.sha256(output)}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
