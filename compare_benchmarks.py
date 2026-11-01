#!/usr/bin/env python3
"""Compare two manifest-bound result artifacts and export JSON/Markdown reports."""
import argparse
import json
from pathlib import Path

from artifact_io import write_json
from comparison_report import compare_results, markdown_report
from dataset_manifest import load_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    inputs = {path.resolve() for path in (args.baseline, args.candidate, args.manifest)}
    outputs = [path.resolve() for path in (args.output, args.markdown) if path]
    if len(outputs) != len(set(outputs)) or inputs.intersection(outputs):
        parser.error("input and output paths must be distinct")
    report = compare_results(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), load_manifest(args.manifest))
    write_json(report, args.output)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report))
    print(f"Compared {report['overall']['sample_count']} pairs; delta={report['overall']['mean_score_delta']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
