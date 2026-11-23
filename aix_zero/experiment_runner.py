"""One recorded candidate experiment using trusted training/evaluation adapters.

Adapters execute in-process. Worker isolation, resource enforcement, automatic
promotion and daemon scheduling are not provided by this orchestration layer.
"""
from copy import deepcopy
import json

from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_splits import verify_manifest, eligible_training_rows
from aix_zero.candidate_training import TrainingOutput
from aix_zero.candidate_policy import screen_candidate, validate_policy


def run_experiment(registry, champion, manifest, metadata, policy, *, identity, train, evaluate):
    manifest = deepcopy(manifest); policy = deepcopy(policy); metadata = deepcopy(metadata)
    validate_policy(policy)
    payload = verify_manifest(manifest)
    training = [r['row'] for r in payload['splits']['train']]
    training = eligible_training_rows(manifest, training, identity)
    if not training:
        raise ValueError('Training split must not be empty')
    if metadata['dataset_sha256'] != manifest['sha256']:
        raise ValueError('Dataset identity differs from manifest')
    # Freeze policy and manifest identity before any training adapter executes.
    frozen = dict(metadata, settings=dict(metadata['settings'], policy=policy))
    attempt = registry.create(champion, frozen)
    directory = registry.root / 'attempts' / attempt
    def publish(name, value):
        path=directory/name
        registry._exclusive(path,json.dumps(value,sort_keys=True,allow_nan=False).encode())
        return path
    publish('split_manifest.json',manifest)
    publish('policy.json',policy)
    registry.transition(attempt,'proposed','running','training started')
    try:
        champion_digest=registry.inspect(attempt)['manifest']['champion_sha256']
        # Pass immutable baseline bytes and only training rows, never held-out answers.
        candidate_bytes=train(registry.verify(champion_digest).read_bytes(),deepcopy(training),deepcopy(frozen))
        training_metrics = None
        if isinstance(candidate_bytes, TrainingOutput):
            training_metrics = candidate_bytes.metrics
            candidate_bytes = candidate_bytes.checkpoint
        if not isinstance(candidate_bytes,bytes) or not candidate_bytes:
            raise ValueError('Trainer must return nonempty checkpoint bytes')
        training_digest = None
        if training_metrics is not None:
            training_digest = registry.snapshot(publish('training.json', training_metrics))
        candidate_path=directory/'candidate.checkpoint'
        registry._exclusive(candidate_path,candidate_bytes)
        candidate_digest=registry.snapshot(candidate_path)
        dev=evaluate(deepcopy(manifest),registry.verify(champion_digest),registry.verify(candidate_digest),split='development')
        reg=evaluate(deepcopy(manifest),registry.verify(champion_digest),registry.verify(candidate_digest),split='regression')
        expected=dict(champion=champion_digest,candidate=candidate_digest)
        for split, report in (('development', dev), ('regression', reg)):
            if report.get('model_identities') != expected or report.get('split_manifest_sha256') != manifest['sha256'] or report.get('tokenizer_sha256') != metadata['tokenizer_sha256']:
                raise ValueError('Evaluation provenance differs from experiment')
            declared_evaluation = frozen['settings'].get('evaluation')
            if declared_evaluation is not None and report.get('settings') != declared_evaluation:
                raise ValueError('Evaluation settings differ from frozen experiment')
            expected_rows = {r['row']['prompt']: r['row']['gold'] for r in payload['splits'][split]}
            for role in ('champion', 'candidate'):
                records = report['arms'][role]['records']
                observed = {r['prompt']: r['gold'] for r in records}
                if len(records) != len(expected_rows) or observed != expected_rows:
                    raise ValueError('Evaluation cases differ from registered split')
        decision=screen_candidate(dev,reg,policy)
        evidence=publish('evaluation.json',dict(development=dev,regression=reg,decision=decision,candidate_sha256=candidate_digest,training_metrics_sha256=training_digest))
        registry.transition(attempt,'running','evaluated','evaluation evidence recorded',evidence)
        if decision['decision']=='reject':
            registry.transition(attempt,'evaluated','rejected','; '.join(decision['reasons']))
        elif decision['decision']=='blocked':
            registry.transition(attempt,'evaluated','failed',decision['reason'])
        return dict(attempt=attempt,state=registry.inspect(attempt)['state'],decision=decision,candidate_sha256=candidate_digest,training_metrics_sha256=training_digest)
    except Exception as error:
        state=registry.inspect(attempt)['state']
        if state in ('running','evaluated'):
            registry.transition(attempt,state,'failed',type(error).__name__+': '+str(error))
        raise
