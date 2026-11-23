"""Predeclared tiny-set fitting and generalization diagnostic check (#424).

Measures whether the pinned neural model and training path can fit a small (2-8 row)
verified single-skill training set across recorded seeds, then distinguishes training-set
reproduction (memorization) from prompt-disjoint same-skill development generalization
and fixed retention. Never replaces the live champion.
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

from aix_zero.candidate_policy import compute_retention_metrics
from aix_zero.candidate_training import validate_training_settings
from aix_zero.experiment_splits import eligible_training_rows, verify_manifest
from aix_zero.inspect_candidate_experiment import compare_paired_cases
from aix_zero.paired_evaluation import evaluate_pair
from aix_zero.prepare_candidate_experiment import build_diagnostic_splits


def _rss_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss) if sys.platform == 'darwin' else int(usage.ru_maxrss) * 1024


def verify_encoding_and_gradient_contract(model, tokenizer, rows):
    """Verify token round-trip, BOS/EOS, shifted labels, nonzero supervised tokens,
    context fit, trainable groups, finite nonzero gradients, and actual parameter changes.
    """
    import torch
    from aix_zero.meta_ai_autopilot import encode_complete_training_batch, verify_r1_reasoning_trace

    roundtrip_checks = []
    for row in rows:
        prompt = row['prompt']
        completion = row['completion']
        if not verify_r1_reasoning_trace(completion, row['gold'])['fully_verified']:
            raise ValueError(f'Row failed trace verification: {prompt}')
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        c_ids = tokenizer.encode(completion, add_bos=False, add_eos=True)
        if not p_ids or p_ids[0] != tokenizer.bos_id:
            raise ValueError('Prompt encoding missing BOS token')
        if not c_ids or c_ids[-1] != tokenizer.eos_id:
            raise ValueError('Completion encoding missing EOS token')
        decoded_prompt = tokenizer.decode(p_ids, skip_special=True)
        decoded_completion = tokenizer.decode(c_ids, skip_special=True)
        if decoded_prompt != prompt or decoded_completion != completion:
            raise ValueError(f'Tokenizer round-trip mismatch for {prompt}')
        roundtrip_checks.append(dict(
            prompt=prompt,
            prompt_tokens=len(p_ids),
            completion_tokens=len(c_ids),
            total_tokens=len(p_ids) + len(c_ids),
            bos_present=True,
            eos_present=True,
            roundtrip_exact=True,
        ))

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
    if encoding['skipped_overlength'] > 0 or encoding['encoded'] != len(rows):
        raise ValueError('Training rows exceeded context limit during contract check')

    shift_labels = labels[:, 1:]
    supervised_tokens = int((shift_labels != 0).sum().item())
    if supervised_tokens <= 0:
        raise ValueError('Zero supervised tokens in shifted labels')

    # Verify shifted labels align with completion tokens (prompt positions masked with 0)
    for idx, info in enumerate(roundtrip_checks):
        p_len = info['prompt_tokens']
        tot_len = info['total_tokens']
        if not (labels[idx, :p_len] == 0).all().item():
            raise ValueError('Prompt tokens were not masked in labels')
        if not (labels[idx, p_len:tot_len] == inputs[idx, p_len:tot_len]).all().item():
            raise ValueError('Completion labels do not match input token IDs')

    work_model = deepcopy(model)
    allowed_groups = ('experts.', 'router.', 'lm_head.', 'decoder_layers.', 'memory_bank.')
    trainable_names = []
    frozen_names = []
    for name, parameter in work_model.named_parameters():
        is_trainable = any(part in name for part in allowed_groups)
        parameter.requires_grad = is_trainable
        if is_trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    frozen_before = {name: p.detach().clone() for name, p in work_model.named_parameters() if not p.requires_grad}
    trainable_before = {name: p.detach().clone() for name, p in work_model.named_parameters() if p.requires_grad}

    work_model.train()
    optimizer = torch.optim.AdamW([p for p in work_model.parameters() if p.requires_grad], lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    loss = work_model(inputs, labels=labels, token_weights=token_weights)['loss']
    if not torch.isfinite(loss).item():
        raise ValueError('Non-finite loss in gradient contract check')
    loss.backward()

    grad_norms = {}
    for name, parameter in work_model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise ValueError(f'Missing or non-finite gradient for {name}')
            grad_norms[name] = float(parameter.grad.detach().norm().item())
        elif parameter.grad is not None and float(parameter.grad.detach().abs().max().item()) != 0.0:
            raise ValueError(f'Frozen parameter {name} received nonzero gradient')

    nonzero_grad_tensors = sum(1 for v in grad_norms.values() if v > 0.0)
    if nonzero_grad_tensors == 0:
        raise ValueError('All trainable gradients were zero')

    torch.nn.utils.clip_grad_norm_([p for p in work_model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()

    for name, parameter in work_model.named_parameters():
        if not parameter.requires_grad:
            if not torch.equal(parameter, frozen_before[name]):
                raise ValueError(f'Frozen parameter {name} changed during step')
    changed_trainable = sum(
        1 for name, parameter in work_model.named_parameters()
        if parameter.requires_grad and not torch.equal(parameter, trainable_before[name])
    )
    if changed_trainable == 0:
        raise ValueError('No trainable parameters changed after optimizer step')

    return dict(
        contract_verified=True,
        example_count=len(rows),
        context_limit=int(model.config.max_seq_len),
        supervised_tokens_per_step=supervised_tokens,
        trainable_tensor_count=len(trainable_names),
        frozen_tensor_count=len(frozen_names),
        trainable_parameter_count=sum(p.numel() for p in work_model.parameters() if p.requires_grad),
        nonzero_gradient_tensor_count=nonzero_grad_tensors,
        changed_trainable_tensor_count=changed_trainable,
        frozen_parameters_unchanged=True,
        roundtrip_checks=roundtrip_checks,
    )


def _evaluate_rows_pair(rows, champion_bytes, candidate_bytes, *, max_new_tokens=36):
    import torch
    from aix_zero.benchmark_baseline import load_v11
    from aix_zero.eval_harness_bridge import score_rational_or_probability
    from aix_zero.meta_ai_autopilot import verify_r1_reasoning_trace

    base = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    models = {}
    tokenizers = {}
    identities = {}
    for role, data in (('champion', champion_bytes), ('candidate', candidate_bytes)):
        model, tokenizer = load_v11(base)
        model.load_state_dict(torch.load(io.BytesIO(data), map_location='cpu', weights_only=True), strict=True)
        model.eval()
        models[role] = model
        tokenizers[role] = tokenizer
        identities[role] = hashlib.sha256(data).hexdigest()

    settings = dict(max_new_tokens=max_new_tokens, num_candidates=1, temperature=0.0, inflight_alu=False)

    def adapter(role):
        def generate(prompt, gen_settings):
            tok = tokenizers[role]
            mdl = models[role]
            ids = tok.encode(prompt, add_bos=True, add_eos=False)
            with torch.no_grad():
                generated, _, _ = mdl.generate_verified(ids, eos_id=tok.eos_id, tokenizer=tok, **gen_settings)
            new_ids = generated[len(ids):]
            if new_ids and new_ids[-1] == tok.eos_id:
                term = 'eos'
            elif len(generated) >= mdl.config.max_seq_len:
                term = 'context_limit'
            elif len(new_ids) >= gen_settings['max_new_tokens']:
                term = 'max_new_tokens'
            else:
                term = 'incomplete'
            return dict(
                completion=tok.decode(new_ids, skip_special=False),
                prompt_tokens=len(ids),
                generated_tokens=len(new_ids),
                total_tokens=len(generated),
                termination_reason=term,
            )
        return generate

    report = evaluate_pair(
        rows,
        adapter('champion'),
        adapter('candidate'),
        identities=identities,
        settings=settings,
        score_answer=lambda text, gold: score_rational_or_probability(text, gold, is_probability=False)['score'] == 1.0,
        score_trace=lambda text, gold: verify_r1_reasoning_trace(text, gold)['fully_verified'],
    )
    report.update(
        assistance_mode='neural_only',
        context_limit=models['champion'].config.max_seq_len,
    )
    return report


def _train_with_loss_curve(initial_bytes, rows, *, seed, steps, learning_rate, max_seconds, tokenizer_sha256):
    import torch
    from aix_zero.benchmark_baseline import load_v11
    from aix_zero.meta_ai_autopilot import encode_complete_training_batch, verify_r1_reasoning_trace

    settings = dict(steps=steps, learning_rate=learning_rate, max_seconds=max_seconds)
    validate_training_settings(settings)
    if not (2 <= len(rows) <= 8):
        raise ValueError('Tiny-set learning check requires 2 to 8 training examples')
    for row in rows:
        if not verify_r1_reasoning_trace(row['completion'], row['gold'])['fully_verified']:
            raise ValueError('Training row failed trusted trace verification')

    start = time.monotonic()

    def check_budget():
        if time.monotonic() - start > max_seconds:
            raise TimeoutError('Cooperative training time budget exceeded')

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
            for name, parameter in model.named_parameters():
                parameter.requires_grad = any(part in name for part in ('experts.', 'router.', 'lm_head.', 'decoder_layers.', 'memory_bank.'))
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

            def compute_loss():
                val = model(inputs, labels=labels, token_weights=token_weights)['loss']
                if not torch.isfinite(val).item():
                    raise ValueError('Non-finite training loss')
                return val

            loss_curve = []
            check_budget()
            model.eval()
            with torch.no_grad():
                before = float(compute_loss())
            loss_curve.append(dict(step=0, loss=before))
            model.train()
            for step_idx in range(1, steps + 1):
                check_budget()
                optimizer.zero_grad(set_to_none=True)
                loss_val = compute_loss()
                loss_val.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True).item())
                optimizer.step()
                loss_curve.append(dict(step=step_idx, loss=float(loss_val.detach().item()), grad_norm=grad_norm))
            check_budget()
            model.eval()
            with torch.no_grad():
                after = float(compute_loss())
            state = model.state_dict()
            stream = io.BytesIO()
            torch.save(state, stream)
            cand_bytes = stream.getvalue()
            sup_per_step = int((labels[:, 1:] != 0).sum().item())
            return cand_bytes, dict(
                seed=seed,
                steps_completed=steps,
                learning_rate=learning_rate,
                train_loss_before=before,
                train_loss_after=after,
                loss_curve=loss_curve,
                supervised_tokens_per_step=sup_per_step,
                total_supervised_token_exposures=sup_per_step * steps,
                trainable_parameters=sum(p.numel() for p in parameters),
                elapsed_seconds=time.monotonic() - start,
                peak_rss_bytes=_rss_bytes(),
                candidate_sha256=hashlib.sha256(cand_bytes).hexdigest(),
            )
    finally:
        torch.set_num_threads(prev_threads)


def run_tiny_learning_check(
    *,
    manifest=None,
    champion_path=None,
    seeds=(42, 43),
    steps=20,
    learning_rate=0.002,
    max_training_seconds=120.0,
    max_new_tokens=36,
    target_train_exact_match_ratio=1.0,
    min_development_net_gain=1,
    min_champion_regression_correct=1,
):
    """Run the predeclared tiny-set learning check across seeds and classify outcome:
    - `fitting_failed`: fails to fit the tiny training set within the predeclared step cap
    - `memorization_only_no_generalization`: fits training set, zero/negative development gain
    - `target_gain_with_regression_loss`: fits and improves development, but loses established skill
    - `fitted_and_generalized_without_regression`: fits training set, positive dev gain, 0 regression losses
    """
    import torch
    from aix_zero.benchmark_baseline import load_v11

    if not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('At least one nonnegative integer seed is required')
    base_safetensors = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    model, tokenizer = load_v11(base_safetensors)
    if champion_path is not None:
        raw_champ = Path(champion_path).read_bytes()
        model.load_state_dict(torch.load(io.BytesIO(raw_champ), map_location='cpu', weights_only=True), strict=True)
        champion_bytes = raw_champ
    else:
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        champion_bytes = stream.getvalue()

    champion_sha256 = hashlib.sha256(champion_bytes).hexdigest()
    if manifest is None:
        manifest = build_diagnostic_splits(tokenizer, model.config.max_seq_len, profile='single_skill_tiny')
    payload = verify_manifest(manifest)

    train_rows = [r['row'] for r in payload['splits']['train']]
    train_rows = eligible_training_rows(manifest, train_rows, lambda r: r['prompt'])
    dev_rows = [r['row'] for r in payload['splits']['development']]
    reg_rows = [r['row'] for r in payload['splits']['regression']]
    if not (2 <= len(train_rows) <= 8):
        raise ValueError('Tiny-set check requires 2 to 8 training examples')
    if not dev_rows or not reg_rows:
        raise ValueError('Development and regression splits must be nonempty')

    contract = verify_encoding_and_gradient_contract(model, tokenizer, train_rows)

    # Unchanged control arm (champion evaluated against itself)
    unchanged_train = _evaluate_rows_pair(train_rows, champion_bytes, champion_bytes, max_new_tokens=max_new_tokens)
    unchanged_dev = _evaluate_rows_pair(dev_rows, champion_bytes, champion_bytes, max_new_tokens=max_new_tokens)
    unchanged_reg = _evaluate_rows_pair(reg_rows, champion_bytes, champion_bytes, max_new_tokens=max_new_tokens)

    seed_runs = []
    failed_attempts = []
    for seed in seeds:
        try:
            cand_bytes, train_metrics = _train_with_loss_curve(
                champion_bytes,
                train_rows,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
                max_seconds=max_training_seconds,
                tokenizer_sha256=tokenizer.vocab_sha256,
            )
            train_eval = _evaluate_rows_pair(train_rows, champion_bytes, cand_bytes, max_new_tokens=max_new_tokens)
            dev_eval = _evaluate_rows_pair(dev_rows, champion_bytes, cand_bytes, max_new_tokens=max_new_tokens)
            reg_eval = _evaluate_rows_pair(reg_rows, champion_bytes, cand_bytes, max_new_tokens=max_new_tokens)
            train_Eval_copy = dict(train_eval, split='train')
            dev_Eval_copy = dict(dev_eval, split='development')
            reg_Eval_copy = dict(reg_eval, split='regression')
            retention = compute_retention_metrics(reg_Eval_copy, min_champion_correct=min_champion_regression_correct)

            train_cand_correct = train_eval['arms']['candidate']['correct']
            train_cand_trace = train_eval['arms']['candidate']['trace_valid']
            train_ratio = train_cand_correct / len(train_rows)
            fitted = train_ratio >= target_train_exact_match_ratio and train_cand_trace == train_cand_correct
            dev_net_gain = dev_eval['paired']['gains'] - dev_eval['paired']['losses']

            seed_runs.append(dict(
                seed=seed,
                status='completed',
                training=train_metrics,
                fitting_check=dict(
                    target_exact_match_ratio=target_train_exact_match_ratio,
                    champion_train_correct=train_eval['arms']['champion']['correct'],
                    candidate_train_correct=train_cand_correct,
                    candidate_train_trace_valid=train_cand_trace,
                    total_train_cases=len(train_rows),
                    candidate_train_exact_match_ratio=train_ratio,
                    fitted_target_met=fitted,
                    train_diagnostics=compare_paired_cases(train_Eval_copy),
                ),
                generalization_check=dict(
                    champion_dev_correct=dev_eval['arms']['champion']['correct'],
                    candidate_dev_correct=dev_eval['arms']['candidate']['correct'],
                    champion_dev_trace_valid=dev_eval['arms']['champion']['trace_valid'],
                    candidate_dev_trace_valid=dev_eval['arms']['candidate']['trace_valid'],
                    total_dev_cases=len(dev_rows),
                    development_net_gain=dev_net_gain,
                    generalized_target_met=dev_net_gain >= min_development_net_gain,
                    development_diagnostics=compare_paired_cases(dev_Eval_copy),
                ),
                retention_check=dict(
                    retention_metrics=retention,
                    regression_diagnostics=compare_paired_cases(reg_Eval_copy),
                ),
            ))
        except Exception as error:
            failed_attempts.append(dict(
                seed=seed,
                status='failed',
                error_type=type(error).__name__,
                error=str(error),
            ))

    all_fitted = bool(seed_runs) and not failed_attempts and all(r['fitting_check']['fitted_target_met'] for r in seed_runs)
    all_generalized = all_fitted and all(r['generalization_check']['generalized_target_met'] for r in seed_runs)
    any_reg_loss = any(r['retention_check']['retention_metrics']['losses'] > 0 for r in seed_runs)

    if not all_fitted:
        conclusion = 'fitting_failed'
    elif not all_generalized:
        conclusion = 'memorization_only_no_generalization'
    elif any_reg_loss:
        conclusion = 'target_gain_with_regression_loss'
    else:
        conclusion = 'fitted_and_generalized_without_regression'

    if hashlib.sha256(champion_bytes).hexdigest() != champion_sha256:
        raise RuntimeError('Champion bytes mutated during tiny-set learning check')

    return dict(
        schema_version=1,
        experiment_type='tiny_set_learning_and_generalization_check',
        independent_final_test=False,
        automatic_promotion=False,
        champion_unchanged=True,
        champion_sha256=champion_sha256,
        tokenizer_sha256=tokenizer.vocab_sha256,
        dataset_sha256=manifest['sha256'],
        dataset_provenance=payload.get('dataset_provenance'),
        predeclared_protocol=dict(
            seeds=list(seeds),
            steps=steps,
            learning_rate=learning_rate,
            max_training_seconds=max_training_seconds,
            max_new_tokens=max_new_tokens,
            target_train_exact_match_ratio=target_train_exact_match_ratio,
            min_development_net_gain=min_development_net_gain,
            min_champion_regression_correct=min_champion_regression_correct,
            assistance_mode='neural_only',
            inflight_alu=False,
            retrieval_enabled=False,
            symbolic_repair_enabled=False,
            peer_help_enabled=False,
        ),
        encoding_and_gradient_contract=contract,
        unchanged_checkpoint_control=dict(
            train_correct=unchanged_train['arms']['champion']['correct'],
            development_correct=unchanged_dev['arms']['champion']['correct'],
            regression_correct=unchanged_reg['arms']['champion']['correct'],
        ),
        completed_seed_count=len(seed_runs),
        failed_attempt_count=len(failed_attempts),
        seed_runs=seed_runs,
        failed_attempts=failed_attempts,
        diagnostic_conclusion=conclusion,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--champion', type=Path, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--max-training-seconds', type=float, default=120.0)
    parser.add_argument('--max-new-tokens', type=int, default=36)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing learning check report')
    manifest = json.loads(args.manifest.read_text()) if args.manifest is not None else None
    report = run_tiny_learning_check(
        manifest=manifest,
        champion_path=args.champion,
        seeds=tuple(args.seeds),
        steps=args.steps,
        learning_rate=args.learning_rate,
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
