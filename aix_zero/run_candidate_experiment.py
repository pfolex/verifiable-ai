"""Run one recorded offline candidate experiment; never replace the live model."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from aix_zero.candidate_policy import validate_policy
from aix_zero.candidate_training import train_checkpoint, validate_training_settings
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_runner import run_experiment
from aix_zero.experiment_splits import verify_manifest
from aix_zero.paired_checkpoint_eval import evaluate_checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--champion', type=Path, required=True, help='v11-compatible torch state_dict file')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--max-training-seconds', type=float, default=120)
    parser.add_argument('--max-new-tokens', type=int, default=36)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    training = dict(steps=args.steps, learning_rate=args.learning_rate, max_seconds=args.max_training_seconds)
    validate_training_settings(training)
    if args.seed < 0 or args.seed >= 2**32 or not 1 <= args.max_new_tokens <= 112:
        parser.error('Invalid seed or generation budget')
    manifest = json.loads(args.manifest.read_text())
    payload = verify_manifest(manifest)
    # This CLI uses exact prompt identities. Semantic family isolation must be
    # established by the manifest producer before invoking it.
    for split, records in payload['splits'].items():
        if split != 'final' and not records:
            raise ValueError('Train/development/regression splits must be nonempty')
        for record in records:
            if record['identity'] != record['row']['prompt']:
                raise ValueError('This runner requires prompt-based split identities')
    policy = json.loads(args.policy.read_text())
    validate_policy(policy)
    import torch
    from aix_zero.benchmark_baseline import load_v11
    source = Path(__file__).resolve().parent
    base = source / 'checkpoints/aix_zero_v11_kip.safetensors'
    _, tokenizer = load_v11(base)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    # Hash source bytes as well: the checkout may contain uncommitted edits.
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}
    metadata = dict(source_revision=revision, tokenizer_sha256=tokenizer.vocab_sha256,
        dataset_sha256=manifest['sha256'], seed=args.seed,
        hardware=platform.platform() + ' / ' + platform.machine(),
        settings=dict(training=training, evaluation=dict(max_new_tokens=args.max_new_tokens, num_candidates=1, temperature=0., inflight_alu=False),
                      runtime_source_sha256=sources, baseline_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                      python_version=platform.python_version(), torch_version=str(torch.__version__)))
    result = run_experiment(ExperimentRegistry(args.registry), args.champion, manifest, metadata, policy,
        identity=lambda row: row['prompt'], train=train_checkpoint,
        evaluate=partial(evaluate_checkpoints, max_new_tokens=args.max_new_tokens))
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
