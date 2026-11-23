"""Prepare portable, validated inputs for offline candidate experiments (#422).

Writes an immutable champion torch state_dict, a verified four-split manifest, and a
frozen screening policy under a user-selected accessible directory, then prints the
exact runnable `python -m aix_zero.run_candidate_experiment` command.
"""
import argparse
import errno
import hashlib
import io
import json
from pathlib import Path
import sys

from aix_zero.candidate_policy import validate_policy
from aix_zero.experiment_splits import build_split_manifest, verify_manifest


# Fixed, versioned retention and champion-success diagnostic rows independently defined
# from the v11 21-domain baseline curriculum (do not depend on candidate outcomes).
FIXED_RETENTION_SUITE = (
    dict(
        domain='CODE_TRACE',
        prompt='Q:Code:def f(x):return 6*x+1;f(2)=',
        completion='<think>6*2=12;12+1=13</think><ans>13</ans>',
        gold='13',
        retention_subset='champion_v11_verified_success',
    ),
    dict(
        domain='PHYSICS_KINEMATICS',
        prompt='Q:Phys:a=3,t=5,u=2,v=',
        completion='<think>3*5=15;15+2=17</think><ans>17</ans>',
        gold='17',
        retention_subset='champion_v11_verified_success',
    ),
    dict(
        domain='META_TRL_RLOO_LEAVE_ONE_OUT',
        prompt='Q:MetaRLOO:r1=1;other_sum=1;k=4->rloo_advantage?',
        completion='<think>base=1/3;adv=1-1/3=2/3</think><ans>2/3</ans>',
        gold='2/3',
        retention_subset='fixed_meta_ai_retention',
    ),
    dict(
        domain='META_SPECULATIVE_DECODING',
        prompt='Q:MetaSpec:accepted=2;window=5->accept_ratio?',
        completion='<think>accept=2/5=2/5</think><ans>2/5</ans>',
        gold='2/5',
        retention_subset='fixed_meta_ai_retention',
    ),
)


def _actionable_io_error(error, path, role):
    code = getattr(error, 'errno', None)
    if isinstance(error, FileNotFoundError) or code == errno.ENOENT:
        return FileNotFoundError(
            f"Missing {role} at '{path}'. Check that the file exists and pass a repo-relative "
            f"or accessible path (avoid literal '/path/to/...' placeholders)."
        )
    if isinstance(error, PermissionError) or code in (errno.EPERM, errno.EACCES):
        return PermissionError(
            f"Access denied for {role} at '{path}' ({error}). On macOS/Linux, protected folders "
            f"such as Documents or root directories may block Python processes; choose a writable "
            f"directory inside your repository checkout or temporary workspace instead of changing "
            f"broad system permissions."
        )
    return error


def build_diagnostic_splits(tokenizer, max_seq_len, *, profile='diagnostic', domain='META_QUANTIZATION_FP8_INT4'):
    """Build a deterministic, template-derived diagnostic split bundle.

    Explicitly marks `independent_final_test = False` and records known template/exposure
    notes so diagnostic sets are never mistaken for unseen final benchmarks.
    """
    from fractions import Fraction
    from aix_zero.meta_ai_autopilot import build_meta_ai_self_improvement_curriculum, verify_r1_reasoning_trace

    def fits_and_verified(row):
        if not verify_r1_reasoning_trace(row['completion'], row['gold'])['fully_verified']:
            return False
        ids = (
            tokenizer.encode(row['prompt'], add_bos=True, add_eos=False)
            + tokenizer.encode(row['completion'], add_bos=False, add_eos=True)
        )
        return len(ids) <= max_seq_len

    splits = {name: [] for name in ('train', 'development', 'regression', 'final')}
    seen = set()

    def add_row(split, row):
        prompt = row['prompt']
        if prompt in seen or not fits_and_verified(row):
            return False
        seen.add(prompt)
        splits[split].append(dict(row))
        return True

    if profile == 'single_skill_tiny':
        # 4 training examples and 4 prompt-disjoint development examples in one skill domain
        train_params = ((32, 4), (32, 8), (64, 4), (64, 8))
        dev_params = ((16, 4), (16, 8), (48, 4), (48, 8))
        for split_name, pairs in (('train', train_params), ('development', dev_params)):
            for fp_bits, q_bits in pairs:
                f_val = Fraction(fp_bits, q_bits)
                gold = str(f_val.numerator) if f_val.denominator == 1 else f'{f_val.numerator}/{f_val.denominator}'
                row = dict(
                    domain='META_QUANTIZATION_FP8_INT4',
                    prompt=f'Q:MetaQuant:fp16_bits={fp_bits};quant_bits={q_bits}->compression?',
                    completion=f'<think>comp={fp_bits}/{q_bits}={gold}</think><ans>{gold}</ans>',
                    gold=gold,
                    template_family='META_QUANTIZATION_FP8_INT4',
                    diagnostic_only=True,
                )
                if not add_row(split_name, row):
                    raise ValueError(f'Diagnostic row failed verification or context limit: {row["prompt"]}')
        for row in FIXED_RETENTION_SUITE:
            if not add_row('regression', row):
                raise ValueError(f'Fixed retention row failed verification or context limit: {row["prompt"]}')
    elif profile == 'diagnostic':
        for split_name, cycle in (('train', 101), ('development', 2100)):
            for row in build_meta_ai_self_improvement_curriculum(cycle_index=cycle):
                if domain and row['domain'] != domain and split_name in ('train', 'development'):
                    continue
                add_row(split_name, row)
                if len(splits[split_name]) >= 4:
                    break
        for row in FIXED_RETENTION_SUITE:
            add_row('regression', row)
        for row in build_meta_ai_self_improvement_curriculum(cycle_index=3100):
            if len(splits['regression']) >= 6:
                break
            add_row('regression', row)
    else:
        raise ValueError(f'Unsupported profile: {profile}')

    if any(not splits[s] for s in ('train', 'development', 'regression')):
        raise ValueError('Prepared train, development, and regression splits must be nonempty')

    manifest = build_split_manifest(splits, lambda row: row['prompt'])
    manifest['payload']['dataset_provenance'] = dict(
        profile=profile,
        independent_final_test=False,
        seeded_diagnostic_templates=True,
        selection_procedure=(
            'Deterministic template-derived diagnostic splits with a fixed, candidate-independent '
            'v11 champion-success retention subset (CODE_TRACE, PHYSICS_KINEMATICS) plus Meta-AI probes.'
        ),
        known_exposure_limitations=(
            'v11 checkpoint was previously trained on the 21-domain KIP curriculum;evolving v12 '
            'checkpoints were exposed to Meta-AI curriculum cycles 1..108. Diagnostic splits measure '
            'controlled fitting and non-vacuous retention, not independent final benchmark mastery.'
        ),
    )
    encoded = json.dumps(manifest['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    manifest['sha256'] = hashlib.sha256(encoded).hexdigest()
    verify_manifest(manifest)
    return manifest


def prepare_experiment_inputs(
    output_dir,
    *,
    source_checkpoint=None,
    profile='single_skill_tiny',
    min_cases=None,
    min_net_gain=1,
    max_regression_losses=0,
    max_trace_losses=0,
    min_champion_regression_correct=1,
    steps=15,
    learning_rate=0.002,
    max_training_seconds=120,
    max_new_tokens=36,
    seed=42,
):
    import torch
    from aix_zero.benchmark_baseline import load_v11
    from aix_zero.experiment_registry import ExperimentRegistry

    base_safetensors = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    source_path = Path(source_checkpoint) if source_checkpoint is not None else base_safetensors
    output_root = Path(output_dir)

    try:
        raw_source = source_path.read_bytes()
    except OSError as error:
        raise _actionable_io_error(error, source_path, 'source checkpoint') from error

    model, tokenizer = load_v11(base_safetensors)
    if source_path.resolve() != base_safetensors.resolve():
        try:
            state = torch.load(io.BytesIO(raw_source), map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise ValueError(f"Incompatible v11 torch state_dict at '{source_path}': {error}") from error

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    champion_bytes = stream.getvalue()
    champion_sha256 = hashlib.sha256(champion_bytes).hexdigest()

    manifest = build_diagnostic_splits(tokenizer, model.config.max_seq_len, profile=profile)
    payload = verify_manifest(manifest)
    effective_min_cases = (
        min_cases
        if min_cases is not None
        else min(len(payload['splits']['development']), len(payload['splits']['regression']))
    )
    policy = dict(
        min_cases=int(effective_min_cases),
        min_net_gain=int(min_net_gain),
        max_regression_losses=int(max_regression_losses),
        max_trace_losses=int(max_trace_losses),
    )
    if min_champion_regression_correct is not None and min_champion_regression_correct > 0:
        policy['min_champion_regression_correct'] = int(min_champion_regression_correct)
    policy_sha256 = validate_policy(policy)

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _actionable_io_error(error, output_root, 'output directory') from error

    champion_dest = output_root / 'champion.pt'
    manifest_dest = output_root / 'split_manifest.json'
    policy_dest = output_root / 'policy.json'
    bundle_dest = output_root / 'bundle.json'
    registry_dest = output_root / 'registry'

    for path in (champion_dest, manifest_dest, policy_dest, bundle_dest):
        if path.exists():
            existing_bundles = {}
            for existing_path in (champion_dest, manifest_dest, policy_dest):
                if existing_path.exists():
                    existing_bundles[existing_path.name] = hashlib.sha256(existing_path.read_bytes()).hexdigest()
            raise FileExistsError(
                f"Refusing to overwrite existing experiment inputs in '{output_root}'. "
                f"Existing file digests: {json.dumps(existing_bundles, sort_keys=True)}"
            )

    try:
        ExperimentRegistry._exclusive(champion_dest, champion_bytes)
        ExperimentRegistry._exclusive(
            manifest_dest,
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode(),
        )
        ExperimentRegistry._exclusive(
            policy_dest,
            json.dumps(policy, indent=2, sort_keys=True, allow_nan=False).encode(),
        )
    except OSError as error:
        raise _actionable_io_error(error, output_root, 'experiment bundle files') from error

    python_exe = sys.executable or 'python3'
    run_command = [
        python_exe,
        '-m',
        'aix_zero.run_candidate_experiment',
        '--champion',
        str(champion_dest),
        '--manifest',
        str(manifest_dest),
        '--policy',
        str(policy_dest),
        '--registry',
        str(registry_dest),
        '--steps',
        str(steps),
        '--learning-rate',
        str(learning_rate),
        '--max-training-seconds',
        str(max_training_seconds),
        '--max-new-tokens',
        str(max_new_tokens),
        '--seed',
        str(seed),
    ]
    import shlex
    shell_command = ' '.join(shlex.quote(part) for part in run_command)

    bundle = dict(
        schema_version=1,
        profile=profile,
        independent_final_test=False,
        seeded_diagnostic_templates=True,
        source_checkpoint=str(source_path),
        source_checkpoint_sha256=hashlib.sha256(raw_source).hexdigest(),
        champion_path=str(champion_dest),
        champion_sha256=champion_sha256,
        manifest_path=str(manifest_dest),
        manifest_sha256=manifest['sha256'],
        policy_path=str(policy_dest),
        policy_sha256=policy_sha256,
        registry_path=str(registry_dest),
        tokenizer_sha256=tokenizer.vocab_sha256,
        split_counts={k: len(payload['splits'][k]) for k in ('train', 'development', 'regression', 'final')},
        run_argv=run_command,
        run_command=shell_command,
    )
    ExperimentRegistry._exclusive(
        bundle_dest,
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False).encode(),
    )
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New or empty writable directory for inputs')
    parser.add_argument('--source-checkpoint', type=Path, default=None,
                        help='Optional v11-compatible torch state_dict or defaults to v11 safetensors')
    parser.add_argument('--profile', choices=('single_skill_tiny', 'diagnostic'), default='single_skill_tiny')
    parser.add_argument('--min-cases', type=int, default=None)
    parser.add_argument('--min-net-gain', type=int, default=1)
    parser.add_argument('--max-regression-losses', type=int, default=0)
    parser.add_argument('--max-trace-losses', type=int, default=0)
    parser.add_argument('--min-champion-regression-correct', type=int, default=1)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--max-training-seconds', type=float, default=120)
    parser.add_argument('--max-new-tokens', type=int, default=36)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    bundle = prepare_experiment_inputs(
        args.output_dir,
        source_checkpoint=args.source_checkpoint,
        profile=args.profile,
        min_cases=args.min_cases,
        min_net_gain=args.min_net_gain,
        max_regression_losses=args.max_regression_losses,
        max_trace_losses=args.max_trace_losses,
        min_champion_regression_correct=args.min_champion_regression_correct,
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_training_seconds=args.max_training_seconds,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False))
    return bundle


if __name__ == '__main__':
    main()
