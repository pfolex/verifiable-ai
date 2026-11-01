#!/usr/bin/env python3
"""Evaluate reference, recorded, Ollama or hosted submissions on public VeriSci tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark_runner import run_benchmark, save_result
from provider_adapters import ReferenceProvider, ReplayProvider
from ollama_provider import OllamaProvider
from dataset_manifest import load_manifest, make_manifest, save_manifest
from hosted_providers import HostedProvider

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


def generate_cases(generation):
    from rlvr_forge import TaskForge
    forge = TaskForge(split="eval")
    cases = []
    for domain in generation["domains"]:
        for offset in range(generation["samples_per_domain"]):
            task = forge.generate(seed=generation["seed"] + offset, domain=domain, difficulty=generation["difficulty"])
            cases.append({"sample_id": task.task_id, "input": task.to_dict()})
    return cases


def main(argv=None):
    from rlvr_forge import TaskForge
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("reference", "replay", "ollama", "openai", "groq"), default="reference")
    parser.add_argument("--recording", type=Path, help="JSON object with model and responses keyed by sample ID")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, help="Rerun an integrity-checked saved public manifest")
    parser.add_argument("--export-manifest", type=Path, help="Save the exact public task manifest")
    parser.add_argument("--manifest-only", action="store_true", help="Export without invoking a provider")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--samples-per-domain", type=int)
    parser.add_argument("--difficulty", type=int, choices=range(1, 11))
    parser.add_argument("--domains", nargs="+", choices=TaskForge.DOMAINS)
    parser.add_argument("--model", help="Explicit model ID, required for live providers")
    parser.add_argument("--ollama-url", help="Ollama HTTP(S) origin, default http://localhost:11434")
    parser.add_argument("--timeout", type=float, help="Live request timeout in seconds, default 120")
    parser.add_argument("--temperature", type=float, help="Live sampling temperature; hosted default omitted, Ollama default 0")
    parser.add_argument("--max-output-tokens", type=int, help="Live generation limit, default 2048")
    parser.add_argument("--thinking", choices=("auto", "on", "off"), help="Ollama thinking control, default auto")
    parser.add_argument("--api-url", help="Hosted API origin override; HTTPS or loopback HTTP for tests")
    parser.add_argument("--pricing", type=Path, help="Explicit versioned hosted-model token price card")
    args = parser.parse_args(argv)
    generation_options = (args.seed, args.difficulty, args.domains, args.samples_per_domain)
    if args.manifest and any(value is not None for value in generation_options):
        parser.error("generation options cannot override a saved manifest")
    if args.manifest_only:
        if not args.export_manifest or args.output or args.provider != "reference":
            parser.error("--manifest-only requires --export-manifest, no --output and no live/replay provider")
    elif not args.output:
        parser.error("--output is required unless exporting a manifest only")
    inputs = [path.resolve() for path in (args.manifest, args.recording, args.pricing) if path]
    outputs = [path.resolve() for path in (args.output, args.export_manifest) if path]
    if len(outputs) != len(set(outputs)) or set(inputs) & set(outputs):
        parser.error("input and output paths must be distinct")
    if (args.provider == "replay") != (args.recording is not None):
        parser.error("--recording is required only with --provider replay")
    live_options = (args.model, args.ollama_url, args.timeout, args.temperature, args.max_output_tokens, args.thinking)
    hosted = args.provider in ("openai", "groq")
    if args.provider not in ("ollama", "openai", "groq") and any(value is not None for value in live_options):
        parser.error("live provider options require a live provider")
    if args.provider in ("ollama", "openai", "groq") and not args.model:
        parser.error("--model is required with live providers")
    if args.provider != "ollama" and (args.ollama_url is not None or args.thinking is not None):
        parser.error("--ollama-url and --thinking require --provider ollama")
    if not hosted and (args.api_url is not None or args.pricing is not None):
        parser.error("--api-url and --pricing require a hosted provider")
    verifier_revision = revision("verisci")
    private_tasks = None
    if args.manifest:
        manifest = load_manifest(args.manifest)
        if manifest["scorer"]["version"] != verifier_revision:
            parser.error("saved manifest requires a different VeriSci scorer revision")
    else:
        generation = {"seed": 2026 if args.seed is None else args.seed,
                      "difficulty": 3 if args.difficulty is None else args.difficulty,
                      "domains": list(TaskForge.DOMAINS) if args.domains is None else args.domains,
                      "samples_per_domain": 2 if args.samples_per_domain is None else args.samples_per_domain}
        if generation["samples_per_domain"] < 1 or len(generation["domains"]) != len(set(generation["domains"])):
            parser.error("require positive samples-per-domain and unique domains")
        cases = generate_cases(generation)
        private_tasks = {case["sample_id"]: case["input"] for case in cases}
        manifest = make_manifest(cases,
            {"name": "rlvr-forge-procedural", "revision": revision("rlvr-forge"), "split": "eval"},
            {"name": "verisci-reward", "version": verifier_revision, "configuration": generation}, generation)
    generation = manifest["generation"]
    cases = manifest["cases"]
    forge_revision = manifest["dataset"]["revision"]
    if args.provider == "reference" and not args.manifest_only and private_tasks is None:
        if forge_revision != revision("rlvr-forge"):
            parser.error("reference reruns require the manifest's RLVR-Forge revision")
        reconstructed = generate_cases(generation)
        expected = make_manifest(reconstructed, manifest["dataset"], manifest["scorer"], generation)
        if expected != manifest:
            parser.error("manifest does not match reconstructed public tasks")
        private_tasks = {case["sample_id"]: case["input"] for case in reconstructed}
    if args.export_manifest:
        save_manifest(manifest, args.export_manifest)
    if args.manifest_only:
        print(f"Saved manifest {args.export_manifest}: {len(cases)} public samples")
        return 0
    if args.provider == "reference":
        provider = ReferenceProvider(forge_revision, private_tasks)
    elif args.provider == "replay":
        provider = ReplayProvider(json.loads(args.recording.read_text()))
        if set(provider.responses) != {case["sample_id"] for case in cases}:
            parser.error("recording response IDs must exactly match generated sample IDs")
    elif args.provider == "ollama":
        try:
            provider = OllamaProvider(
                args.model, base_url=args.ollama_url or "http://localhost:11434",
                timeout=120 if args.timeout is None else args.timeout,
                temperature=0 if args.temperature is None else args.temperature,
                num_predict=2048 if args.max_output_tokens is None else args.max_output_tokens,
                seed=generation["seed"],
                think={"on": True, "off": False}.get(args.thinking),
            )
        except Exception as exc:
            parser.error(f"Ollama setup failed ({type(exc).__name__}); check URL, installed model and options")
    else:
        try:
            provider = HostedProvider(
                args.provider, args.model, base_url=args.api_url,
                timeout=120 if args.timeout is None else args.timeout,
                max_output_tokens=2048 if args.max_output_tokens is None else args.max_output_tokens,
                temperature=args.temperature,
                pricing=json.loads(args.pricing.read_text()) if args.pricing else None,
            )
        except Exception as exc:
            parser.error(f"Hosted setup failed ({type(exc).__name__}); check API key environment, model and options")
    result = run_benchmark(
        cases, dataset=manifest["dataset"], provider=provider, scorer=verisci_score,
        scorer_metadata=manifest["scorer"], manifest=manifest,
    )
    save_result(result, args.output)
    failures = sum("error" in sample for sample in result["samples"])
    print(f"Saved {args.output}: {len(cases)} samples, mean={result['summary']['mean_score']:.6f}, errors={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
