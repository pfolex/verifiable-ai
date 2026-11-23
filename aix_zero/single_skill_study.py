"""Predeclared single-skill training recipe comparison (#425) and versioned training replay (#412).

Compares an unchanged champion baseline against at most two predeclared candidate recipes
across multiple recorded seeds with fresh optimizers, complete-target encoding, compute/token
accounting, and non-vacuous retention evaluation. Never replaces the live champion or tunes
on reserved final evaluation splits.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time

from aix_zero.candidate_policy import compute_retention_metrics, screen_candidate, validate_policy
from aix_zero.candidate_training import validate_training_settings
from aix_zero.experiment_splits import build_split_manifest, eligible_training_rows, verify_manifest
from aix_zero.prepare_candidate_experiment import FIXED_RETENTION_SUITE, build_diagnostic_splits
from aix_zero.tiny_learning_check import _evaluate_rows_pair


# Versioned training-only replay anchor pool (distinct from FIXED_RETENTION_SUITE evaluation prompts!)
# These rows are registered inside the `train` split of the manifest so replay never ingests
# evaluation answers or unregistered rows (#407 / #412).
TRAINING_ONLY_REPLAY_POOL = (
    dict(
        domain='CODE_TRACE',
        prompt='Q:Code:def f(x):return 5*x+2;f(3)=',
        completion='<think>5*3=15;15+2=17</think><ans>17</ans>',
        gold='17',
        replay_origin='v11_training_replay_anchor',
        is_replay_anchor=True,
    ),
    dict(
        domain='PHYSICS_KINEMATICS',
        prompt='Q:Phys:a=4,t=3,u=5,v=',
        completion='<think>4*3=12;12+5=17</think><ans>17</ans>',
        gold='17',
        replay_origin='v11_training_replay_anchor',
        is_replay_anchor=True,
    ),
    dict(
        domain='META_TRL_RLOO_LEAVE_ONE_OUT',
        prompt='Q:MetaRLOO:r1=1;other_sum=1;k=6->rloo_advantage?',
        completion='<think>base=1/5;adv=1-1/5=4/5</think><ans>4/5</ans>',
        gold='4/5',
        replay_origin='meta_ai_training_replay_anchor',
        is_replay_anchor=True,
    ),
    dict(
        domain='META_SPECULATIVE_DECODING',
        prompt='Q:MetaSpec:accepted=2;window=6->accept_ratio?',
        completion='<think>accept=2/6=1/3</think><ans>1/3</ans>',
        gold='1/3',
        replay_origin='meta_ai_training_replay_anchor',
        is_replay_anchor=True,
    ),
)


def _rss_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss) if sys.platform == 'darwin' else int(usage.ru_maxrss) * 1024


def build_single_skill_study_manifest(tokenizer, max_seq_len):
    """Build a versioned manifest where `train` includes both target-skill training rows and
    training-only replay anchors, while `development` and `regression` remain strictly disjoint.
    """
    base_manifest = build_diagnostic_splits(tokenizer, max_seq_len, profile='single_skill_tiny')
    base_payload = verify_manifest(base_manifest)
    splits = {
        split: [deepcopy(r['row']) for r in base_payload['splits'][split]]
        for split in ('train', 'development', 'regression', 'final')
    }
    eval_prompts = {r['prompt'] for s in ('development', 'regression', 'final') for r in splits[s]}
    for anchor in TRAINING_ONLY_REPLAY_POOL:
        if anchor['prompt'] in eval_prompts:
            raise ValueError(f'Replay anchor overlaps evaluation split: {anchor["prompt"]}')
        splits['train'].append(deepcopy(anchor))
    manifest = build_split_manifest(splits, lambda r: r['prompt'])
    manifest['payload']['dataset_provenance'] = dict(
        profile='single_skill_study_with_training_replay_pool',
        independent_final_test=False,
        seeded_diagnostic_templates=True,
        target_skill_domain='META_QUANTIZATION_FP8_INT4',
        training_target_count=4,
        training_replay_anchor_count=len(TRAINING_ONLY_REPLAY_POOL),
        selection_procedure=(
            'Predeclared single-skill study manifest combining 4 target-skill training rows '
            'and 4 training-only replay anchors in `train`, evaluated on 4 prompt-disjoint '
            'development rows and 4 fixed retention rows.'
        ),
    )
    encoded = json.dumps(manifest['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    manifest['sha256'] = hashlib.sha256(encoded).hexdigest()
    verify_manifest(manifest)
    return manifest


def sample_registered_replay_mixture(manifest, *, include_replay=True, replay_limit=2, seed=42):
    """Select target training rows and optional training-only replay rows strictly from
    the registered `train` split of `manifest`, enforcing `eligible_training_rows` (#407 / #412).
    """
    import random
    payload = verify_manifest(manifest)
    all_train = [r['row'] for r in payload['splits']['train']]
    # Verify every candidate training row through the fail-closed ingestion guard
    verified_train = eligible_training_rows(manifest, all_train, lambda r: r['prompt'])
    target_rows = [r for r in verified_train if not r.get('is_replay_anchor')]
    replay_pool = [r for r in verified_train if r.get('is_replay_anchor')]
    if not target_rows:
        raise ValueError('No target-skill training rows in manifest')

    selected_replay = []
    if include_replay:
        if not replay_pool:
            raise ValueError('Replay requested but manifest has no registered replay anchors')
        rng = random.Random(seed)
        ordered_pool = list(replay_pool)
        rng.shuffle(ordered_pool)
        selected_replay = ordered_pool[:max(1, min(len(ordered_pool), int(replay_limit)))]

    combined = target_rows + selected_replay
    # Re-verify combined mixture through fail-closed eligibility gate
    eligible = eligible_training_rows(manifest, combined, lambda r: r['prompt'])
    return dict(
        rows=eligible,
        target_count=len(target_rows),
        replay_count=len(selected_replay),
        total_count=len(eligible),
        replay_prompts=[r['prompt'] for r in selected_replay],
        mixture_seed=seed,
    )


DEFAULT_PREDECLARED_RECIPES = (
    dict(
        recipe_id='recipe_a_target_only_lr1e3',
        hypothesis='Target-skill training without replay fits target rows rapidly but risks losing established retention skills.',
        steps=10,
        learning_rate=0.001,
        include_replay=False,
        replay_limit=0,
        trainable_scope='full_adapter_groups',
    ),
    dict(
        recipe_id='recipe_b_replay_balanced_lr1e3',
        hypothesis='Mixing registered training-only replay anchors with target-skill rows at matched step count preserves established retention skills.',
        steps=10,
        learning_rate=0.001,
        include_replay=True,
        replay_limit=2,
        trainable_scope='full_adapter_groups',
    ),
)


def _train_recipe_arm(initial_bytes, rows, *, seed, steps, learning_rate, max_seconds, tokenizer_sha256, trainable_scope='full_adapter_groups'):
    import torch
    from aix_zero.benchmark_baseline import load_v11
    from aix_zero.meta_ai_autopilot import encode_complete_training_batch, verify_r1_reasoning_trace

    validate_training_settings(dict(steps=steps, learning_rate=learning_rate, max_seconds=max_seconds))
    for row in rows:
        if not verify_r1_reasoning_trace(row['completion'], row['gold'])['fully_verified']:
            raise ValueError('Training row failed trusted trace verification')

    start = time.monotonic()
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            base = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
            model, tokenizer = load_v11(base)
            if tokenizer.vocab_sha256 != tokenizer_sha256:
                raise ValueError('Tokenizer identity mismatch')
            model.load_state_dict(torch.load(io.BytesIO(initial_bytes), map_location='cpu', weights_only=True), strict=True)
            weights = {
                tokenizer.ans_id,
                tokenizer.end_ans_id,
                *(tokenizer.token_to_id[ch] for ch in '0123456789*/=;-+' if ch in tokenizer.token_to_id),
            }
            inputs, labels, token_weights, encoding = encode_complete_training_batch(
                rows,
                tokenizer,
                weights,
                model.config.max_seq_len,
            )
            if encoding['skipped_overlength']:
                raise ValueError('Registered training rows exceed model context')

            if trainable_scope == 'single_routed_expert':
                allowed = ('experts.2.',)
            elif trainable_scope == 'full_adapter_groups':
                allowed = ('experts.', 'router.', 'lm_head.', 'decoder_layers.', 'memory_bank.')
            else:
                raise ValueError(f'Unsupported trainable_scope: {trainable_scope}')

            for name, parameter in model.named_parameters():
                parameter.requires_grad = any(part in name for part in allowed)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

            def loss_fn():
                val = model(inputs, labels=labels, token_weights=token_weights)['loss']
                if not torch.isfinite(val).item():
                    raise ValueError('Non-finite training loss')
                return val

            model.eval()
            with torch.no_grad():
                before = float(loss_fn())
            model.train()
            for _ in range(steps):
                if time.monotonic() - start > max_seconds:
                    raise TimeoutError('Cooperative training time budget exceeded')
                optimizer.zero_grad(set_to_none=True)
                loss_fn().backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                after = float(loss_fn())
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            cand_bytes = stream.getvalue()
            sup_per_step = int((labels[:, 1:] != 0).sum().item())
            return cand_bytes, dict(
                seed=seed,
                steps_completed=steps,
                learning_rate=learning_rate,
                trainable_scope=trainable_scope,
                trainable_parameters=sum(p.numel() for p in parameters),
                supervised_tokens_per_step=sup_per_step,
                total_supervised_tokens=sup_per_step * steps,
                train_loss_before=before,
                train_loss_after=after,
                elapsed_seconds=time.monotonic() - start,
                peak_rss_bytes=_rss_bytes(),
                candidate_sha256=hashlib.sha256(cand_bytes).hexdigest(),
            )
    finally:
        torch.set_num_threads(prev_threads)


def run_predeclared_training_study(
    *,
    manifest=None,
    champion_path=None,
    recipes=DEFAULT_PREDECLARED_RECIPES,
    seeds=(42, 43, 44),
    max_training_seconds=120.0,
    max_new_tokens=36,
    policy=None,
):
    """Run the predeclared single-skill recipe comparison across >= 3 seeds (#425).

    Compares unchanged baseline and at most 2 candidate recipes from identical champion bytes.
    Reports every seed, failed runs, paired answer/trace outcomes, retention metrics, and
    compute matching basis (equal optimizer steps vs total supervised tokens).
    """
    import torch
    from aix_zero.benchmark_baseline import load_v11

    if not (1 <= len(recipes) <= 2):
        raise ValueError('Predeclared study permits at most two candidate recipes alongside unchanged baseline')
    if len(seeds) < 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Predeclared study requires at least three nonnegative integer seeds')

    base_safetensors = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    model, tokenizer = load_v11(base_safetensors)
    if champion_path is not None:
        champion_bytes = Path(champion_path).read_bytes()
        model.load_state_dict(torch.load(io.BytesIO(champion_bytes), map_location='cpu', weights_only=True), strict=True)
    else:
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        champion_bytes = stream.getvalue()
    champion_sha256 = hashlib.sha256(champion_bytes).hexdigest()

    if manifest is None:
        manifest = build_single_skill_study_manifest(tokenizer, model.config.max_seq_len)
    payload = verify_manifest(manifest)

    target_train_rows = [r['row'] for r in payload['splits']['train'] if not r['row'].get('is_replay_anchor')]
    dev_rows = [r['row'] for r in payload['splits']['development']]
    reg_rows = [r['row'] for r in payload['splits']['regression']]

    if policy is None:
        policy = dict(
            min_cases=min(len(dev_rows), len(reg_rows)),
            min_net_gain=1,
            max_regression_losses=0,
            max_trace_losses=0,
            min_champion_regression_correct=1,
        )
    policy_sha256 = validate_policy(policy)

    recipe_matrix = dict(
        schema_version=1,
        champion_sha256=champion_sha256,
        tokenizer_sha256=tokenizer.vocab_sha256,
        dataset_sha256=manifest['sha256'],
        policy=deepcopy(policy),
        policy_sha256=policy_sha256,
        seeds=list(seeds),
        max_new_tokens=max_new_tokens,
        max_training_seconds=max_training_seconds,
        recipes=deepcopy(list(recipes)),
    )
    recipe_manifest_sha256 = hashlib.sha256(
        json.dumps(recipe_matrix, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()

    # Evaluate unchanged baseline control once (deterministic greedy temperature=0.0)
    base_train = _evaluate_rows_pair(target_train_rows, champion_bytes, champion_bytes, max_new_tokens=max_new_tokens)
    base_dev = _evaluate_rows_pair(dev_rows, champion_bytes, champion_bytes, max_new_tokens=max_new_tokens)
    base_reg = _evaluate_rows_pair(reg_rows, champion_bytes, champion_bytes, max_new_tokens=max_new_tokens)
    base_retention = compute_retention_metrics(
        dict(base_reg, split='regression'),
        min_champion_correct=policy.get('min_champion_regression_correct', 0),
    )

    recipe_reports = []
    for recipe in recipes:
        rid = recipe['recipe_id']
        seed_results = []
        failures = []
        for seed in seeds:
            try:
                mixture = sample_registered_replay_mixture(
                    manifest,
                    include_replay=bool(recipe.get('include_replay', False)),
                    replay_limit=int(recipe.get('replay_limit', 0)),
                    seed=seed,
                )
                cand_bytes, metrics = _train_recipe_arm(
                    champion_bytes,
                    mixture['rows'],
                    seed=seed,
                    steps=int(recipe['steps']),
                    learning_rate=float(recipe['learning_rate']),
                    max_seconds=max_training_seconds,
                    tokenizer_sha256=tokenizer.vocab_sha256,
                    trainable_scope=recipe.get('trainable_scope', 'full_adapter_groups'),
                )
                tr_eval = _evaluate_rows_pair(target_train_rows, champion_bytes, cand_bytes, max_new_tokens=max_new_tokens)
                dv_eval = _evaluate_rows_pair(dev_rows, champion_bytes, cand_bytes, max_new_tokens=max_new_tokens)
                rg_eval = _evaluate_rows_pair(reg_rows, champion_bytes, cand_bytes, max_new_tokens=max_new_tokens)
                dv_report = dict(
                    dv_eval,
                    split='development',
                    split_manifest_sha256=manifest['sha256'],
                    tokenizer_sha256=tokenizer.vocab_sha256,
                )
                rg_report = dict(
                    rg_eval,
                    split='regression',
                    split_manifest_sha256=manifest['sha256'],
                    tokenizer_sha256=tokenizer.vocab_sha256,
                )
                decision = screen_candidate(dv_report, rg_report, policy)
                seed_results.append(dict(
                    seed=seed,
                    status='completed',
                    mixture={k: v for k, v in mixture.items() if k != 'rows'},
                    training=metrics,
                    target_train_outcomes=dict(
                        champion_correct=tr_eval['arms']['champion']['correct'],
                        candidate_correct=tr_eval['arms']['candidate']['correct'],
                        candidate_trace_valid=tr_eval['arms']['candidate']['trace_valid'],
                        total=len(target_train_rows),
                    ),
                    development_outcomes=dict(
                        champion_correct=dv_eval['arms']['champion']['correct'],
                        candidate_correct=dv_eval['arms']['candidate']['correct'],
                        champion_trace_valid=dv_eval['arms']['champion']['trace_valid'],
                        candidate_trace_valid=dv_eval['arms']['candidate']['trace_valid'],
                        paired=dv_eval['paired'],
                        net_gain=decision['development_net_gain'],
                    ),
                    regression_outcomes=dict(
                        champion_correct=rg_eval['arms']['champion']['correct'],
                        candidate_correct=rg_eval['arms']['candidate']['correct'],
                        champion_trace_valid=rg_eval['arms']['champion']['trace_valid'],
                        candidate_trace_valid=rg_eval['arms']['candidate']['trace_valid'],
                        paired=rg_eval['paired'],
                        retention=decision['retention'],
                    ),
                    screening_decision=decision,
                ))
            except Exception as error:
                failures.append(dict(
                    seed=seed,
                    status='failed',
                    error_type=type(error).__name__,
                    error=str(error),
                ))

        eligible_seeds = sum(
            1 for r in seed_results if r['screening_decision']['decision'] == 'eligible_for_review'
        )
        replicated_pass = (len(seed_results) == len(seeds)) and (eligible_seeds == len(seeds))
        mean_dev_gain = (
            sum(r['development_outcomes']['net_gain'] for r in seed_results) / len(seed_results)
            if seed_results else None
        )
        mean_reg_losses = (
            sum(r['regression_outcomes']['retention']['losses'] for r in seed_results) / len(seed_results)
            if seed_results else None
        )
        mean_train_correct = (
            sum(r['target_train_outcomes']['candidate_correct'] for r in seed_results) / len(seed_results)
            if seed_results else None
        )
        recipe_reports.append(dict(
            recipe_id=rid,
            hypothesis=recipe['hypothesis'],
            recipe_settings=deepcopy(recipe),
            completed_seed_count=len(seed_results),
            failed_seed_count=len(failures),
            eligible_seed_count=eligible_seeds,
            replicated_improvement_across_all_seeds=replicated_pass,
            aggregate_metrics=dict(
                mean_target_train_correct=mean_train_correct,
                mean_development_net_gain=mean_dev_gain,
                mean_regression_losses=mean_reg_losses,
                supervised_tokens_per_seed=(
                    seed_results[0]['training']['total_supervised_tokens'] if seed_results else None
                ),
                optimizer_steps_per_seed=int(recipe['steps']),
            ),
            seed_results=seed_results,
            failures=failures,
        ))

    any_replicated = any(r['replicated_improvement_across_all_seeds'] for r in recipe_reports)
    if hashlib.sha256(champion_bytes).hexdigest() != champion_sha256:
        raise RuntimeError('Champion bytes mutated during predeclared training study')

    return dict(
        schema_version=1,
        study_type='predeclared_single_skill_training_comparison',
        exploratory_status='predeclared_diagnostic_comparison_not_final_test',
        independent_final_test=False,
        automatic_promotion=False,
        champion_unchanged=True,
        recipe_manifest_sha256=recipe_manifest_sha256,
        recipe_matrix=recipe_matrix,
        compute_matching_basis=(
            'Matched optimizer steps (10 steps/seed) and identical starting champion bytes; '
            'supervised token counts are reported explicitly per recipe because adding replay '
            'anchors increases tokens per step.'
        ),
        unchanged_baseline_control=dict(
            target_train_correct=base_train['arms']['champion']['correct'],
            development_correct=base_dev['arms']['champion']['correct'],
            regression_correct=base_reg['arms']['champion']['correct'],
            retention=base_retention,
        ),
        recipe_reports=recipe_reports,
        study_decision=(
            'proceed_to_independent_confirmation'
            if any_replicated
            else 'stop_and_revise_no_recipe_replicated_improvement'
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--champion', type=Path, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--max-training-seconds', type=float, default=120.0)
    parser.add_argument('--max-new-tokens', type=int, default=36)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing training study report')
    manifest = json.loads(args.manifest.read_text()) if args.manifest is not None else None
    report = run_predeclared_training_study(
        manifest=manifest,
        champion_path=args.champion,
        seeds=tuple(args.seeds),
        max_training_seconds=args.max_training_seconds,
        max_new_tokens=args.max_new_tokens,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
