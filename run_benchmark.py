#!/usr/bin/env python3
"""Evaluate local reference or recorded provider submissions on procedural VeriSci tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark_runner import run_benchmark, save_result
from provider_adapters import ReferenceProvider, ReplayProvider

ROOT = Path(__file__).resolve().parent
for repo in ("verisci", "rlvr-forge"):
    sys.path.insert(0, str(ROOT / repo / "src"))


def revision(repo):
    """Record source identity, including dirty tracked source state."""
    head = subprocess.check_output(["git", "-C", str(ROOT / repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT / repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    new_source = subprocess.check_output(
        ["git", "-C", str(ROOT / repo), "ls-files", "--others", "--exclude-standard", "*.py"], text=True)
    if dirty or new_source:
        raise ValueError(f"{repo} must have a clean working tree for reproducible provenance")
    return head


def verisci_score(task, output):
    from verisci import DomainTask, verify_submission
    submission = json.loads(output)
    if not isinstance(submission, dict):
        raise ValueError("submission must be a JSON object")
    return verify_submission(DomainTask(**task), submission).reward


def main(argv=None):
    from rlvr_forge import TaskForge
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("reference", "replay"), default="reference")
    parser.add_argument("--recording", type=Path, help="JSON object with model and responses keyed by sample ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-domain", type=int, default=2)
    parser.add_argument("--difficulty", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--domains", nargs="+", choices=TaskForge.DOMAINS, default=list(TaskForge.DOMAINS))
    args = parser.parse_args(argv)
    if args.samples_per_domain < 1 or len(args.domains) != len(set(args.domains)):
        parser.error("require positive samples-per-domain and unique domains")
    if (args.provider == "replay") != (args.recording is not None):
        parser.error("--recording is required only with --provider replay")
    forge_revision = revision("rlvr-forge")
    verifier_revision = revision("verisci")
    forge = TaskForge(split="eval")
    cases = []
    for domain in args.domains:
        for offset in range(args.samples_per_domain):
            task = forge.generate(seed=args.seed + offset, domain=domain, difficulty=args.difficulty)
            cases.append({"sample_id": task.task_id, "input": task.to_dict()})
    if args.provider == "reference":
        provider = ReferenceProvider(forge_revision)
    else:
        provider = ReplayProvider(json.loads(args.recording.read_text()))
        if set(provider.responses) != {case["sample_id"] for case in cases}:
            parser.error("recording response IDs must exactly match generated sample IDs")
    result = run_benchmark(
        cases, dataset={"name": "rlvr-forge-procedural", "revision": forge_revision, "split": "eval"},
        provider=provider, scorer=verisci_score,
        scorer_metadata={"name": "verisci-reward", "version": verifier_revision,
                         "configuration": {"seed": args.seed, "difficulty": args.difficulty,
                                           "domains": args.domains, "samples_per_domain": args.samples_per_domain}},
    )
    save_result(result, args.output)
    failures = sum("error" in sample for sample in result["samples"])
    print(f"Saved {args.output}: {len(cases)} samples, mean={result['summary']['mean_score']:.6f}, errors={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
