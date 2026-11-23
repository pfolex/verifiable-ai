"""Case-level failure taxonomy, attempt inspection, and decoding budget probe (#423, #426)."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from aix_zero.candidate_policy import compute_retention_metrics, screen_candidate, validate_policy
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_splits import verify_manifest
from aix_zero.paired_checkpoint_eval import evaluate_checkpoints


def classify_case_failure(record):
    """Classify a single evaluated record into an explicit failure taxonomy."""
    if not record.get('scored', False):
        return dict(
            failure_kind='generation_error',
            answer_status='unscored_error',
            trace_status='unscored_error',
            termination_reason=record.get('termination_reason', 'error'),
            generated_tokens=record.get('generated_tokens'),
            error_type=record.get('error_type'),
            error=record.get('error'),
        )
    from aix_zero.eval_harness_bridge import score_rational_or_probability
    from aix_zero.meta_ai_autopilot import verify_r1_reasoning_trace
    completion = record.get('completion', '')
    gold = str(record.get('gold', ''))
    ans_check = score_rational_or_probability(completion, gold, is_probability=False)
    trace_check = verify_r1_reasoning_trace(completion, gold)

    term = record.get('termination_reason')
    if term is None:
        if completion.endswith('<eos>'):
            term = 'eos'
        elif not completion.strip():
            term = 'empty_output'
        else:
            term = 'not_recorded_no_eos'

    has_ans_open = '<ans>' in completion
    has_ans_close = '</ans>' in completion
    if record.get('answer_correct'):
        answer_status = 'correct_answer'
    elif not completion.strip() or completion.strip() == '<eos>':
        answer_status = 'empty_output'
    elif not (has_ans_open and has_ans_close) or completion.find('<ans>') > completion.find('</ans>'):
        answer_status = 'missing_or_unclosed_answer_tag'
    elif ans_check['status'] in ('MALFORMED_FRACTION', 'ZERO_DENOMINATOR'):
        answer_status = 'malformed_answer'
    else:
        answer_status = 'wrong_answer'

    if record.get('trace_valid'):
        trace_status = 'valid_trace'
    elif not trace_check['format_ok']:
        trace_status = 'malformed_or_missing_think_ans_tags'
    elif not trace_check['equations_ok']:
        trace_status = 'invalid_intermediate_calculation'
    elif not trace_check['answer_ok']:
        trace_status = 'trace_format_and_equations_ok_but_wrong_final_answer'
    else:
        trace_status = 'unverified_trace'

    if record.get('answer_correct') and record.get('trace_valid'):
        failure_kind = 'none'
    elif record.get('answer_correct') and not record.get('trace_valid'):
        failure_kind = 'correct_answer_invalid_trace'
    elif answer_status in ('empty_output', 'missing_or_unclosed_answer_tag', 'malformed_answer'):
        failure_kind = answer_status
    elif not trace_check['equations_ok'] and trace_check['format_ok']:
        failure_kind = 'invalid_calculation_and_wrong_answer'
    else:
        failure_kind = 'wrong_answer'

    return dict(
        failure_kind=failure_kind,
        answer_status=answer_status,
        trace_status=trace_status,
        termination_reason=term,
        generated_tokens=record.get('generated_tokens'),
        checked_equations=trace_check['checked_equations'],
        failed_equations=trace_check['failed_equations'],
        parsed_fraction=ans_check['parsed_fraction'],
    )


def summarize_arm_diagnostics(arm):
    records = arm['records']
    classified = [dict(prompt=r['prompt'], gold=r['gold'], domain=r.get('domain', 'unspecified'),
                       completion=r.get('completion', ''), answer_correct=r.get('answer_correct', False),
                       trace_valid=r.get('trace_valid', False), **classify_case_failure(r)) for r in records]
    failure_counts = {}
    termination_counts = {}
    by_domain = {}
    for item in classified:
        fk = item['failure_kind']
        failure_counts[fk] = failure_counts.get(fk, 0) + 1
        tr = item['termination_reason']
        termination_counts[tr] = termination_counts.get(tr, 0) + 1
        dom = item['domain']
        d_entry = by_domain.setdefault(dom, dict(offered=0, answer_correct=0, trace_valid=0,
                                                 correct_answer_invalid_trace=0, failures=0))
        d_entry['offered'] += 1
        if item['answer_correct']:
            d_entry['answer_correct'] += 1
        if item['trace_valid']:
            d_entry['trace_valid'] += 1
        if fk == 'correct_answer_invalid_trace':
            d_entry['correct_answer_invalid_trace'] += 1
        if fk != 'none':
            d_entry['failures'] += 1
    return dict(
        complete=arm.get('complete', False),
        offered=arm['offered'],
        scored=arm['scored'],
        answer_correct=arm['correct'],
        trace_valid=arm['trace_valid'],
        correct_answer_invalid_trace=failure_counts.get('correct_answer_invalid_trace', 0),
        failure_kind_counts=failure_counts,
        termination_reason_counts=termination_counts,
        per_domain=by_domain,
        cases=classified,
    )


def compare_paired_cases(report):
    champ_arm = summarize_arm_diagnostics(report['arms']['champion'])
    cand_arm = summarize_arm_diagnostics(report['arms']['candidate'])
    champ_by_p = {c['prompt']: c for c in champ_arm['cases']}
    cand_by_p = {c['prompt']: c for c in cand_arm['cases']}
    answer_gains = []
    answer_losses = []
    trace_gains = []
    trace_losses = []
    for prompt, a in champ_by_p.items():
        b = cand_by_p[prompt]
        delta_entry = dict(
            prompt=prompt,
            domain=a['domain'],
            gold=a['gold'],
            champion_completion=a['completion'],
            candidate_completion=b['completion'],
            champion_answer_correct=a['answer_correct'],
            candidate_answer_correct=b['answer_correct'],
            champion_trace_valid=a['trace_valid'],
            candidate_trace_valid=b['trace_valid'],
            champion_failure_kind=a['failure_kind'],
            candidate_failure_kind=b['failure_kind'],
            champion_termination_reason=a['termination_reason'],
            candidate_termination_reason=b['termination_reason'],
        )
        if not a['answer_correct'] and b['answer_correct']:
            answer_gains.append(delta_entry)
        elif a['answer_correct'] and not b['answer_correct']:
            answer_losses.append(delta_entry)
        if not a['trace_valid'] and b['trace_valid']:
            trace_gains.append(delta_entry)
        elif a['trace_valid'] and not b['trace_valid']:
            trace_losses.append(delta_entry)
    return dict(
        split=report.get('split'),
        comparison_complete=report.get('comparison_complete', False),
        settings=deepcopy(report.get('settings', {})),
        context_limit=report.get('context_limit'),
        paired_summary=deepcopy(report.get('paired')),
        champion=champ_arm,
        candidate=cand_arm,
        answer_gains=answer_gains,
        answer_losses=answer_losses,
        trace_gains=trace_gains,
        trace_losses=trace_losses,
    )


def inspect_attempt(registry_root, attempt):
    """Read-only attempt inspection that verifies all artifact digests and explains case-level outcomes."""
    registry = ExperimentRegistry(registry_root)
    inspected = registry.inspect(attempt)
    directory = registry.root / 'attempts' / attempt
    manifest_data = inspected['manifest']
    split_manifest = json.loads((directory / 'split_manifest.json').read_text())
    verify_manifest(split_manifest)
    if split_manifest['sha256'] != manifest_data['metadata']['dataset_sha256']:
        raise ValueError('Split manifest digest does not match attempt metadata')
    policy = json.loads((directory / 'policy.json').read_text())
    validate_policy(policy)

    eval_path = directory / 'evaluation.json'
    training_path = directory / 'training.json'
    training_metrics = None
    if training_path.exists():
        training_metrics = json.loads(training_path.read_text())

    if not eval_path.exists():
        return dict(
            attempt=attempt,
            state=inspected['state'],
            events=inspected['events'],
            champion_sha256=manifest_data['champion_sha256'],
            dataset_sha256=split_manifest['sha256'],
            training_metrics=training_metrics,
            evaluation_present=False,
        )

    evaluation = json.loads(eval_path.read_text())
    candidate_sha = evaluation['candidate_sha256']
    registry.verify(candidate_sha)
    if evaluation.get('training_metrics_sha256'):
        verified_train_path = registry.verify(evaluation['training_metrics_sha256'])
        training_metrics = json.loads(verified_train_path.read_text())

    dev_diag = compare_paired_cases(evaluation['development'])
    reg_diag = compare_paired_cases(evaluation['regression'])
    min_champ = policy.get('min_champion_regression_correct', 0)
    retention = compute_retention_metrics(evaluation['regression'], min_champion_correct=min_champ)
    recomputed_decision = screen_candidate(evaluation['development'], evaluation['regression'], policy)

    return dict(
        attempt=attempt,
        state=inspected['state'],
        events=inspected['events'],
        champion_sha256=manifest_data['champion_sha256'],
        candidate_sha256=candidate_sha,
        dataset_sha256=split_manifest['sha256'],
        tokenizer_sha256=manifest_data['metadata']['tokenizer_sha256'],
        seed=manifest_data['metadata']['seed'],
        policy=policy,
        training_metrics=training_metrics,
        evaluation_present=True,
        recorded_decision=evaluation['decision'],
        verified_decision=recomputed_decision,
        retention=retention,
        development=dev_diag,
        regression=reg_diag,
    )


def run_decoding_budget_probe(registry_root, attempt, *, probe_max_new_tokens=64, split='development'):
    """Compare the attempt's recorded decoding budget against a larger context-compatible budget.

    Does not mutate the attempt directory or champion; marks result as an exploratory diagnostic probe.
    """
    if split not in ('development', 'regression'):
        raise ValueError('Probe split must be development or regression')
    registry = ExperimentRegistry(registry_root)
    inspected = inspect_attempt(registry_root, attempt)
    if not inspected.get('evaluation_present'):
        raise ValueError('Attempt has no evaluation evidence to probe')
    directory = registry.root / 'attempts' / attempt
    split_manifest = json.loads((directory / 'split_manifest.json').read_text())
    champion_path = registry.verify(inspected['champion_sha256'])
    candidate_path = registry.verify(inspected['candidate_sha256'])

    baseline_split = inspected[split]
    baseline_budget = baseline_split['settings'].get('max_new_tokens', 36)
    probe_report = evaluate_checkpoints(
        split_manifest,
        champion_path,
        candidate_path,
        split=split,
        max_new_tokens=probe_max_new_tokens,
    )
    probe_diag = compare_paired_cases(probe_report)
    return dict(
        schema_version=1,
        probe_type='adaptive_decoding_budget_diagnostic',
        independent_final_test=False,
        eligible_for_promotion=False,
        attempt=attempt,
        split=split,
        champion_sha256=inspected['champion_sha256'],
        candidate_sha256=inspected['candidate_sha256'],
        recorded_budget=dict(
            max_new_tokens=baseline_budget,
            champion_correct=baseline_split['champion']['answer_correct'],
            candidate_correct=baseline_split['candidate']['answer_correct'],
            champion_trace_valid=baseline_split['champion']['trace_valid'],
            candidate_trace_valid=baseline_split['candidate']['trace_valid'],
            champion_termination_counts=baseline_split['champion']['termination_reason_counts'],
            candidate_termination_counts=baseline_split['candidate']['termination_reason_counts'],
            answer_losses=baseline_split['answer_losses'],
            trace_losses=baseline_split['trace_losses'],
        ),
        probed_budget=dict(
            max_new_tokens=probe_max_new_tokens,
            champion_correct=probe_diag['champion']['answer_correct'],
            candidate_correct=probe_diag['candidate']['answer_correct'],
            champion_trace_valid=probe_diag['champion']['trace_valid'],
            candidate_trace_valid=probe_diag['candidate']['trace_valid'],
            champion_termination_counts=probe_diag['champion']['termination_reason_counts'],
            candidate_termination_counts=probe_diag['candidate']['termination_reason_counts'],
            answer_losses=probe_diag['answer_losses'],
            trace_losses=probe_diag['trace_losses'],
        ),
        full_probe_report=probe_report,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--probe-max-new-tokens', type=int, default=None,
                        help='Optional larger generation budget for diagnostic comparison')
    parser.add_argument('--probe-split', choices=('development', 'regression'), default='development')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing inspection report')
    report = inspect_attempt(args.registry, args.attempt)
    if args.probe_max_new_tokens is not None:
        report['decoding_budget_probe'] = run_decoding_budget_probe(
            args.registry,
            args.attempt,
            probe_max_new_tokens=args.probe_max_new_tokens,
            split=args.probe_split,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
