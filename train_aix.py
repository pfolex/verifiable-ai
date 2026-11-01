#!/usr/bin/env python3
"""Self-service AIX experiment: prepare, baseline, causal-LM LoRA, candidate, report."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from artifact_io import canonical_json, write_json
from aix_training_data import checkpoint_identity, file_sha256, load_bundle, prepare_examples
from dataset_manifest import save_manifest
from comparison_report import compare_results, markdown_report
from statistical_gate import assess_comparison, validate_policy

ROOT = Path(__file__).resolve().parent


def default_base():
    return Path(os.environ.get('AIX_BASE_MODEL', ROOT.parent / 'model_merging' / 'merged_r1_aix3_verifiable'))


def prepare(args):
    base, root = args.base_model.resolve(), args.run_dir.resolve()
    if base == root or base in root.parents or root in base.parents:
        raise ValueError('base checkpoint and experiment directories must be separate')
    if root.exists():
        raise ValueError('experiment directory already exists; choose a fresh --run-dir')
    base_config = json.loads((base / 'config.json').read_text())
    if not 1 <= args.last_n_layers <= base_config['num_hidden_layers']:
        raise ValueError('--last-n-layers does not fit the model')
    for name in ('steps', 'rank', 'train_samples_per_domain', 'eval_samples_per_domain', 'max_seq_length', 'max_new_tokens', 'attempts'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or len(set(args.domains)) != len(args.domains):
        raise ValueError('learning rate must be finite and positive; domains must be unique')
    if args.steps < len(args.domains):
        raise ValueError('--steps must cover every requested training domain')
    if args.max_seq_length > base_config['max_position_embeddings'] or args.max_new_tokens >= base_config['max_position_embeddings']:
        raise ValueError('training/generation limits exceed the model context')
    examples, manifest, audit = prepare_examples(domains=args.domains, train_count=args.train_samples_per_domain,
        train_seed=args.train_seed, eval_count=args.eval_samples_per_domain, eval_seed=args.eval_seed,
        eval_difficulty=args.eval_difficulty, attempts=args.attempts)
    policy = json.loads(args.policy.read_text())
    validate_policy(policy)
    if policy['independent_samples'] and audit['evaluation_unique_problems'] != audit['evaluation_samples']:
        raise ValueError('repeated public evaluation problems cannot be declared independent samples')
    identity = checkpoint_identity(base)
    config = {'experiment_version': '1.0.0', 'base_model': str(base), 'base_identity': identity,
              'manifest_sha256': manifest['manifest_sha256'],
              'training': {'steps': args.steps, 'rank': args.rank, 'last_n_layers': args.last_n_layers,
                           'max_seq_length': args.max_seq_length, 'learning_rate': args.learning_rate,
                           'seed': args.train_seed, 'device': args.device, 'gradient_checkpointing': args.gradient_checkpointing},
              'generation': {'max_new_tokens': args.max_new_tokens, 'do_sample': False,
                             'prompt_format': 'aix-causal-json-1.0.0', 'close_thinking_prefill': True},
              'data_audit': audit, 'policy': policy,
              'code_sha256': {name: file_sha256(ROOT / name) for name in (
                  'train_aix.py', 'aix_training_data.py', 'aix_training_runtime.py', 'run_benchmark.py',
                  'benchmark_runner.py', 'evaluation_result.py', 'comparison_report.py', 'statistical_gate.py',
                  'dataset_manifest.py', 'provider_adapters.py', 'artifact_io.py', 'ollama_provider.py',
                  'requirements-training.txt')}}
    root.mkdir(parents=True, exist_ok=False)
    (root / 'train.jsonl').write_text(''.join(canonical_json(example) + '\n' for example in examples))
    config['training_data_sha256'] = file_sha256(root / 'train.jsonl')
    save_manifest(manifest, root / 'eval-manifest.json')
    write_json(config, root / 'experiment.json')
    print(f"Prepared {root}: {len(examples)} unique verified training examples; {audit['evaluation_samples']} evaluation samples ({audit['evaluation_unique_problems']} unique problems)")
    print(f"Training counts: {audit['training_unique_by_domain']}. Exact public-problem overlap: 0.")
    return config


def report(run_dir):
    from aix_training_runtime import experiment_hash
    root = Path(run_dir)
    config, _, manifest = load_bundle(root)
    baseline = json.loads((root / 'baseline.json').read_text())
    candidate = json.loads((root / 'candidate.json').read_text())
    for result in (baseline, candidate):
        parameters = result['model']['parameters']
        if parameters['experiment_sha256'] != experiment_hash(config) or parameters['base_sha256'] != config['base_identity']['sha256']:
            raise ValueError('measurements are from a different experiment')
    for key in ('generation', 'device', 'dtype', 'versions', 'rope_theta'):
        if baseline['model']['parameters'][key] != candidate['model']['parameters'][key]:
            raise ValueError('baseline/candidate execution settings differ')
    if baseline['model']['parameters']['adapter_sha256'] is not None:
        raise ValueError('baseline must be the unadapted checkpoint')
    training = json.loads((root / 'training-report.json').read_text())
    identity = checkpoint_identity(root / 'adapter', adapter=True)
    if (training['experiment_sha256'] != experiment_hash(config) or identity != training['adapter_identity']
            or candidate['model']['parameters']['adapter_sha256'] != identity['sha256']):
        raise ValueError('candidate measurement does not match the saved trained adapter')
    comparison = compare_results(baseline, candidate, manifest)
    gate = assess_comparison(comparison, config['policy'])
    write_json(comparison, root / 'comparison.json')
    (root / 'comparison.md').write_text(markdown_report(comparison))
    write_json(gate, root / 'gate.json')
    print(f"Baseline={comparison['overall']['baseline']['mean_score']:.6f}; candidate={comparison['overall']['candidate']['mean_score']:.6f}; delta={comparison['overall']['mean_score_delta']:+.6f}; gate={gate['decision']}")
    print(f"Read {root / 'comparison.md'}. Completion does not imply improvement or promotion.")
    return gate


def doctor(base):
    from aix_training_runtime import libraries, tokenizer_for, config_for
    torch, transformers, peft = libraries()
    print(f'Training dependencies: torch={torch.__version__}, transformers={transformers.__version__}, peft={peft.__version__}')
    print(f'Hardware: CUDA={torch.cuda.is_available()}, MPS={torch.backends.mps.is_available()} (otherwise CPU; large models can be slow)')
    base = base.resolve()
    config = json.loads((base / 'config.json').read_text())
    from safetensors import safe_open
    tokenizer = tokenizer_for({'base_model': str(base)})
    compatible = config_for({'base_model': str(base)})
    if tokenizer.eos_token_id is None or max(tokenizer.get_vocab().values()) >= config['vocab_size']:
        raise ValueError('tokenizer and checkpoint vocabulary do not match')
    weights = sorted(base.glob('model*.safetensors'))
    if not weights:
        raise ValueError('use a Hugging Face safetensors checkpoint, not GGUF')
    found_embedding = False
    for path in weights:
        with safe_open(path, framework='pt', device='cpu') as handle:
            if 'model.embed_tokens.weight' in handle.keys():
                found_embedding = True
                if handle.get_slice('model.embed_tokens.weight').get_shape() != [config['vocab_size'], config['hidden_size']]:
                    raise ValueError('checkpoint embedding shape does not match config')
    if not found_embedding:
        raise ValueError('expected an AIX/Qwen2 checkpoint with model.embed_tokens.weight')
    total_bytes = sum(path.stat().st_size for path in weights)
    print(f'Checkpoint: {base}; layers={config["num_hidden_layers"]}; weights on disk={total_bytes / 1e9:.2f} GB')
    print(f'Native chat template loaded; RoPE theta={compatible.rope_theta}; EOS token={tokenizer.eos_token_id}')
    print('CPU training uses float32 weights plus activations/optimizer memory. The base will be hashed before an experiment.')


def parser_for():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    d = commands.add_parser('doctor', help='Check dependencies, hardware and tokenizer without loading full weights')
    d.add_argument('--base-model', type=Path, default=default_base())
    for name in ('prepare', 'run'):
        p = commands.add_parser(name, help='Prepare a fresh experiment' if name == 'prepare' else 'Run the complete isolated experiment')
        p.add_argument('--base-model', type=Path, default=default_base())
        p.add_argument('--run-dir', type=Path, required=True)
        p.add_argument('--domains', nargs='+', choices=('logic', 'silicon', 'systems', 'matter'), default=['logic', 'silicon', 'systems', 'matter'])
        p.add_argument('--train-samples-per-domain', type=int, default=16)
        p.add_argument('--eval-samples-per-domain', type=int, default=2)
        p.add_argument('--train-seed', type=int, default=10000)
        p.add_argument('--eval-seed', type=int, default=2026)
        p.add_argument('--eval-difficulty', type=int, choices=range(1, 11), default=3)
        p.add_argument('--attempts', type=int, default=512, help='Maximum generated training candidates per domain')
        p.add_argument('--steps', type=int, default=20)
        p.add_argument('--learning-rate', type=float, default=0.0001)
        p.add_argument('--rank', type=int, default=8)
        p.add_argument('--last-n-layers', type=int, default=2)
        p.add_argument('--max-seq-length', type=int, default=1024)
        p.add_argument('--max-new-tokens', type=int, default=1024)
        p.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
        p.add_argument('--gradient-checkpointing', action='store_true')
        p.add_argument('--policy', type=Path, default=ROOT / 'examples/regression-policy-v1.json')
    for name in ('train', 'evaluate', 'report'):
        p = commands.add_parser(name)
        p.add_argument('--run-dir', type=Path, required=True)
        if name == 'evaluate':
            p.add_argument('--which', choices=('baseline', 'candidate'), required=True)
    return parser


def main(argv=None):
    args = parser_for().parse_args(argv)
    try:
        if args.command == 'doctor':
            doctor(args.base_model)
        elif args.command in ('prepare', 'run'):
            prepare(args)
            if args.command == 'run':
                # Separate processes release the full checkpoint between stages.
                for command in (['evaluate', '--which', 'baseline'], ['train'], ['evaluate', '--which', 'candidate'], ['report']):
                    subprocess.run([sys.executable, str(ROOT / 'train_aix.py'), *command, '--run-dir', str(args.run_dir)], check=True)
        elif args.command == 'train':
            from aix_training_runtime import train_adapter
            train_adapter(args.run_dir)
        elif args.command == 'evaluate':
            from aix_training_runtime import evaluate_checkpoint
            evaluate_checkpoint(args.run_dir, args.which)
        else:
            report(args.run_dir)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Experiment stopped: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
