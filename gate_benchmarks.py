#!/usr/bin/env python3
"""Gate authoritative paired result artifacts. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 3 invalid input."""
import argparse
import json
from pathlib import Path

from artifact_io import write_json
from comparison_report import compare_results
from dataset_manifest import load_manifest
from statistical_gate import assess_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in (args.baseline, args.candidate, args.manifest, args.policy)}:
        parser.error("input and output paths must be distinct")
    try:
        comparison = compare_results(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), load_manifest(args.manifest))
        report = assess_comparison(comparison, json.loads(args.policy.read_text()))
        write_json(report, args.output)
    except Exception as exc:
        print(f"Invalid gate inputs or output failure ({type(exc).__name__})")
        return 3
    print(f"Gate: {report['decision']}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report["decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
