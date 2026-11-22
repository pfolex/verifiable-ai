"""Paired diagnostic evaluation. No statistical or promotion claims are implied."""
from copy import deepcopy
import hashlib
import json
import time


def evaluate_pair(rows, champion, candidate, *, identities, settings, score_answer, score_trace):
    """Generators receive only prompt and settings; reference access belongs to scoring.

    Callers supply neural-only generation adapters and trusted scorers. This API is
    not a sandbox: process isolation and model identity verification remain runner duties.
    """
    cases = deepcopy(list(rows))
    if not cases:
        raise ValueError('Evaluation set must not be empty')
    if set(identities) != {'champion', 'candidate'} or any(not isinstance(v, str) or not v.strip() for v in identities.values()):
        raise ValueError('Both model identities are required')
    prompts = [r['prompt'] for r in cases]
    if any(not isinstance(p, str) or not p for p in prompts) or len(set(prompts)) != len(prompts):
        raise ValueError('Prompts must be unique and nonempty')
    if any('gold' not in r for r in cases):
        raise ValueError('Reference answers are required')
    settings = deepcopy(settings)
    payload = dict(rows=cases, settings=settings)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = dict(schema_version=1, evaluation_sha256=digest, model_identities=deepcopy(identities),
                  settings=settings, assistance_mode='caller_declared_neural_only',
                  independent_final_test=False, statistical_status='NOT_EVALUATED', arms={})
    for name, generate in (('champion', champion), ('candidate', candidate)):
        records=[]
        for row in cases:
            start=time.monotonic()
            record=dict(prompt=row['prompt'], gold=row['gold'], domain=row.get('domain', 'unspecified'))
            try:
                completion=generate(row['prompt'], deepcopy(settings))
                if not isinstance(completion, str):
                    raise TypeError('Generator must return text')
                record['completion']=completion
                # Reference answers are introduced only after generation returns.
                record['answer_correct']=bool(score_answer(completion, row['gold']))
                record['trace_valid']=bool(score_trace(completion, row['gold']))
                record['scored']=True
            except Exception as error:
                record.update(scored=False, answer_correct=False, trace_valid=False,
                              error_type=type(error).__name__, error=str(error))
            record['elapsed_seconds']=time.monotonic()-start
            records.append(record)
        complete=all(r['scored'] for r in records)
        output['arms'][name]=dict(complete=complete, offered=len(cases), scored=sum(r['scored'] for r in records),
                                 correct=sum(r['answer_correct'] for r in records),
                                 trace_valid=sum(r['trace_valid'] for r in records), records=records)
    output['comparison_complete']=all(a['complete'] for a in output['arms'].values())
    output['paired']=None
    if output['comparison_complete']:
        pairs=list(zip(output['arms']['champion']['records'],output['arms']['candidate']['records']))
        output['paired']=dict(gains=sum(not a['answer_correct'] and b['answer_correct'] for a,b in pairs),
                              losses=sum(a['answer_correct'] and not b['answer_correct'] for a,b in pairs),
                              both_correct=sum(a['answer_correct'] and b['answer_correct'] for a,b in pairs),
                              both_wrong=sum(not a['answer_correct'] and not b['answer_correct'] for a,b in pairs))
    return output
