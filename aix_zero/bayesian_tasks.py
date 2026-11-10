"""Exact binary Bayesian tasks; probability tuples and layouts are held out.

p=P(H), a=P(E=1|H), b=P(E=1|not H). Sequential observations are explicitly
conditionally IID given H. All probabilities and answers use exact fractions.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import random
import re


TASKS = ('BP', 'BS')
PROBABILITIES = tuple(Fraction(v) for v in ('1/5', '1/3', '1/2', '2/3', '4/5'))
POLICY = dict(min_test_groups=24, min_improvement_over_best_control=.10,
              min_novel_layout_accuracy=.25, max_fallback_trace_changes=0,
              decision_type='preregistered_empirical_screen_not_SPRT',
              note='Paired layouts/observations are not independent statistical samples.')


def fraction_text(value):
    value = Fraction(value)
    return f'{value.numerator}/{value.denominator}'


def parse_prompt(prompt):
    match = re.fullmatch(r'Q:(BP|BS):([a-z0-9/=;]+);P=', prompt)
    if not match:
        raise ValueError('Unsupported Bayesian prompt grammar')
    task, body = match.groups()
    fields = {}
    for part in body.split(';'):
        key, separator, value = part.partition('=')
        if not separator or key in fields:
            raise ValueError('Invalid or duplicate field')
        fields[key] = value
    if set(fields) != {'p', 'a', 'b', 'e'}:
        raise ValueError('Expected p, a, b, e exactly once')
    values = []
    for key in ('p', 'a', 'b'):
        if not re.fullmatch(r'\d+/[1-9]\d*', fields[key]):
            raise ValueError('Probabilities must be nonnegative rational literals')
        value = Fraction(fields[key])
        if not 0 <= value <= 1:
            raise ValueError('Probability outside [0,1]')
        values.append(value)
    observations = tuple(int(c) for c in fields['e'])
    if not observations or any(c not in (0, 1) for c in observations):
        raise ValueError('Observations must be binary')
    if (len(observations) != 1 and task == 'BP') or not 1 <= len(observations) <= 3:
        raise ValueError('Unsupported observation length')
    return task, *values, observations


def reference_joint(p, a, b, observations):
    """Independent ground truth: enumerate the binary joint distribution."""
    p, a, b = map(Fraction, (p, a, b))
    if not all(0 <= v <= 1 for v in (p, a, b)) or not 1 <= len(observations) <= 3:
        raise ValueError('Invalid probability model')
    joint = {}
    for hypothesis in (0, 1):
        prior = p if hypothesis else 1 - p
        rate = a if hypothesis else b
        for outcome in itertools.product((0, 1), repeat=len(observations)):
            mass = prior
            for bit in outcome:
                mass *= rate if bit else 1 - rate
            joint[hypothesis, outcome] = mass
    assert sum(joint.values()) == 1
    observations = tuple(observations)
    if any(bit not in (0, 1) for bit in observations):
        raise ValueError('Nonbinary observation')
    denominator = sum(joint[h, observations] for h in (0, 1))
    if denominator == 0:
        raise ValueError('Conditioning on an impossible event')
    return joint[1, observations] / denominator


def solve_prompt(prompt):
    """A separate assisted solver using sequential Bayes updates, not the joint oracle."""
    _, posterior, a, b, observations = parse_prompt(prompt)
    for bit in observations:
        lh, ln = (a, b) if bit else (1 - a, 1 - b)
        normalizer = posterior * lh + (1 - posterior) * ln
        if not normalizer:
            raise ValueError('Conditioning on an impossible event')
        posterior = posterior * lh / normalizer
    return fraction_text(posterior)


def render(task, p, a, b, observations, layout):
    fields = dict(p=fraction_text(p), a=fraction_text(a), b=fraction_text(b),
                  e=''.join(map(str, observations)))
    orders = dict(train_a=('p', 'a', 'b', 'e'), train_b=('a', 'p', 'b', 'e'),
                  novel=('e', 'b', 'p', 'a'))
    return f'Q:{task}:' + ';'.join(f'{k}={fields[k]}' for k in orders[layout]) + ';P='


def build_dataset(seed=42):
    groups = list(itertools.product(PROBABILITIES, repeat=3))
    random.Random(seed).shuffle(groups)
    split_groups = dict(train=groups[:85], validation=groups[85:101], test=groups[101:])
    result = dict(schema_version=1, seed=seed, policy=POLICY, tasks={})
    for task in TASKS:
        splits = {}
        for split, tuples in split_groups.items():
            rows = []
            for p, a, b in tuples:
                group = '|'.join(map(fraction_text, (p, a, b)))
                observations = [(0,), (1,)] if task == 'BP' else [(0, 0), (0, 1), (1, 0), (1, 1)]
                if split == 'test' and task == 'BS':
                    observations = [(1, 0), (1, 1)]
                layouts = ('train_a', 'train_b') if split == 'train' else (
                    ('train_a',) if split == 'validation' else ('train_a', 'novel'))
                for obs in observations:
                    for layout in layouts:
                        prompt = render(task, p, a, b, obs, layout)
                        gold = fraction_text(reference_joint(p, a, b, obs))
                        assert solve_prompt(prompt) == gold
                        rows.append(dict(task=task, group=group, layout=layout, prompt=prompt,
                                         gold=gold, generalization='new_values' if layout != 'novel' else 'new_layout'))
                if split == 'test' and task == 'BS':
                    for obs in ((1, 1, 1), (1, 0, 1)):
                        prompt = render(task, p, a, b, obs, 'novel')
                        rows.append(dict(task=task, group=group, layout='novel', prompt=prompt,
                            gold=fraction_text(reference_joint(p, a, b, obs)), generalization='longer_sequence'))
            splits[split] = rows
        splits['majority_answer'] = Counter(r['gold'] for r in splits['train']).most_common(1)[0][0]
        result['tasks'][task] = splits
    return result


def score_answer(text, gold):
    match = re.fullmatch(r'\s*(\d+(?:/[1-9]\d*)?)\s*', text)
    return bool(match and Fraction(match[1]) == Fraction(gold))


def empirical_gate(candidate, controls, *, test_groups, fallback_changes):
    """An engineering screen; repeated layouts are not treated as independent trials."""
    best = max(control['accuracy'] for control in controls.values())
    reasons = []
    if test_groups < POLICY['min_test_groups']:
        reasons.append('insufficient_unique_probability_groups')
    if candidate['accuracy'] - best < POLICY['min_improvement_over_best_control'] - 1e-12:
        reasons.append('insufficient_improvement_over_best_control')
    if candidate['novel_layout_accuracy'] < POLICY['min_novel_layout_accuracy']:
        reasons.append('novel_layout_accuracy_below_threshold')
    if fallback_changes > POLICY['max_fallback_trace_changes']:
        reasons.append('fallback_regression')
    return dict(decision='FAIL' if reasons else 'PASS', reasons=reasons, policy=POLICY,
                best_control_accuracy=best, improvement=candidate['accuracy'] - best,
                unique_test_groups=test_groups, fallback_trace_changes=fallback_changes)
