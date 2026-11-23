"""Deterministic diagnostic screening; never changes a champion checkpoint.

Passing these gates is not statistical significance or independent confirmation.
The caller must freeze the policy before training and enforce artifact provenance.
"""
import hashlib
import json


def _paired_rows(report, split):
    if report.get('split') != split or report.get('comparison_complete') is not True:
        raise ValueError('Missing or incomplete evaluation split')
    if report.get('assistance_mode') != 'neural_only':
        raise ValueError('Only verified neural-only evaluations are eligible')
    arms = report['arms']
    result = {}
    for role in ('champion', 'candidate'):
        arm = arms[role]; rows = arm['records']
        if arm.get('complete') is not True or not rows or arm['offered'] != len(rows) or arm['scored'] != len(rows):
            raise ValueError('Incomplete evaluation arm')
        indexed = {}
        for row in rows:
            if row.get('scored') is not True or type(row.get('answer_correct')) is not bool or type(row.get('trace_valid')) is not bool:
                raise ValueError('Invalid per-case evidence')
            key = row['prompt']
            if not isinstance(key, str) or not key or key in indexed:
                raise ValueError('Invalid prompt identity')
            indexed[key] = row
        result[role] = indexed
    a,b = result['champion'],result['candidate']
    if a.keys() != b.keys() or any(a[k]['gold'] != b[k]['gold'] for k in a):
        raise ValueError('Unpaired questions or references')
    return a,b


def validate_policy(policy):
    required = {'min_cases', 'min_net_gain', 'max_regression_losses', 'max_trace_losses'}
    optional = {'min_champion_regression_correct'}
    if not required.issubset(policy) or set(policy) - (required | optional) or any(type(v) is not int or v < 0 for v in policy.values()):
        raise ValueError('Invalid policy')
    if policy['min_cases'] < 1 or policy['min_net_gain'] < 1:
        raise ValueError('Positive case and gain thresholds required')
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def compute_retention_metrics(regression, *, min_champion_correct=0):
    if type(min_champion_correct) is not int or min_champion_correct < 0:
        raise ValueError('min_champion_correct must be a nonnegative integer')
    champ_rows, cand_rows = _paired_rows(regression, 'regression')
    total_cases = len(champ_rows)
    champion_correct = sum(r['answer_correct'] for r in champ_rows.values())
    candidate_correct = sum(r['answer_correct'] for r in cand_rows.values())
    retained_correct = sum(champ_rows[k]['answer_correct'] and cand_rows[k]['answer_correct'] for k in champ_rows)
    losses = sum(champ_rows[k]['answer_correct'] and not cand_rows[k]['answer_correct'] for k in champ_rows)
    gains = sum(not champ_rows[k]['answer_correct'] and cand_rows[k]['answer_correct'] for k in champ_rows)
    both_wrong = sum(not champ_rows[k]['answer_correct'] and not cand_rows[k]['answer_correct'] for k in champ_rows)
    denominator = champion_correct
    measurable = denominator > 0
    retention_rate = (retained_correct / denominator) if measurable else None
    coverage_met = denominator >= min_champion_correct and (measurable or min_champion_correct == 0)
    if not measurable:
        status = 'not_measurable_zero_champion_successes'
    elif denominator < min_champion_correct:
        status = 'insufficient_champion_coverage'
    elif losses == 0:
        status = 'retained'
    else:
        status = 'regressed'
    return dict(
        total_cases=total_cases,
        champion_correct=champion_correct,
        candidate_correct=candidate_correct,
        retained_champion_correct=retained_correct,
        losses=losses,
        gains=gains,
        both_wrong=both_wrong,
        retention_denominator=denominator,
        retention_measurable=measurable,
        retention_rate=retention_rate,
        zero_losses_non_informative=not measurable,
        min_champion_correct_required=min_champion_correct,
        champion_coverage_sufficient=denominator >= min_champion_correct and (measurable if min_champion_correct > 0 else True),
        retention_status=status,
    )


def screen_candidate(development, regression, policy):
    digest = validate_policy(policy)
    try:
        dev = _paired_rows(development,'development'); reg = _paired_rows(regression,'regression')
        identities = development['model_identities']
        if set(identities) != {'champion','candidate'} or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in identities.values()):
            raise ValueError('Missing checkpoint digests')
        if regression['model_identities'] != identities or identities['champion']==identities['candidate']:
            raise ValueError('Incompatible candidate identities')
        for key in ('settings','tokenizer_sha256','split_manifest_sha256'):
            if key not in development or regression.get(key) != development[key]:
                raise ValueError('Incompatible evaluation configuration')
        if dev[0].keys() & reg[0].keys():
            raise ValueError('Development/regression overlap')
    except (KeyError, TypeError, ValueError) as error:
        return dict(decision='blocked',reason=str(error),policy_sha256=digest)
    net = sum(b['answer_correct']-dev[0][k]['answer_correct'] for k,b in dev[1].items())
    losses = sum(a['answer_correct'] and not reg[1][k]['answer_correct'] for k,a in reg[0].items())
    trace_losses = sum(a['trace_valid'] and not b[k]['trace_valid'] for arows,b in (dev,reg) for k,a in arows.items())
    min_champ = policy.get('min_champion_regression_correct', 0)
    retention = compute_retention_metrics(regression, min_champion_correct=min_champ)
    reasons=[]
    if min(len(dev[0]),len(reg[0])) < policy['min_cases']:reasons.append('insufficient cases')
    if net < policy['min_net_gain']:reasons.append('insufficient development gain')
    if losses > policy['max_regression_losses']:reasons.append('regression losses exceed limit')
    if trace_losses > policy['max_trace_losses']:reasons.append('trace losses exceed limit')
    if min_champ > 0 and not retention['champion_coverage_sufficient']:
        reasons.append('insufficient champion regression coverage')
    if reasons:
        if reasons == ['insufficient champion regression coverage']:
            decision = 'inconclusive_insufficient_retention_evidence'
        else:
            decision = 'reject'
    else:
        decision = 'eligible_for_review'
    return dict(decision=decision, reasons=reasons,
                development_net_gain=net,regression_losses=losses,trace_losses=trace_losses,
                retention=retention,
                policy_sha256=digest,statistical_status='NOT_EVALUATED',automatic_promotion=False)
