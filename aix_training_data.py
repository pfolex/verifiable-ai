"""Verified SFT examples, exact-content holdout checks and checkpoint identities."""
import hashlib
import json
from pathlib import Path

from artifact_io import canonical_json
from dataset_manifest import make_manifest, validate_manifest
from ollama_provider import public_task, SUBMISSION_INSTRUCTIONS
from run_benchmark import ROOT, generate_cases, revision


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path, adapter=False):
    path = Path(path).resolve()
    if adapter:
        names = ['adapter_config.json', 'adapter_model.safetensors']
    else:
        names = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        names += [p.name for p in sorted(path.glob('model*.safetensors'))]
        if not any(name.endswith('.safetensors') for name in names):
            raise ValueError('Use a local Hugging Face safetensors checkpoint, not an Ollama/GGUF file')
        for optional in ('chat_template.jinja', 'model.safetensors.index.json', 'special_tokens_map.json', 'added_tokens.json'):
            if (path / optional).exists():
                names.append(optional)
    files = {name: file_sha256(path / name) for name in names}
    return {'sha256': hashlib.sha256(canonical_json(files).encode()).hexdigest(), 'files': files}


def problem_hash(task):
    public = public_task(task)
    # IDs, split labels and seeds are not evidence of a different problem.
    return hashlib.sha256(canonical_json({k: public[k] for k in ('domain', 'prompt', 'spec')}).encode()).hexdigest()


def messages_for(task):
    public = public_task(task)
    return [{'role': 'system', 'content': 'Return only a JSON submission, without markdown. ' + SUBMISSION_INSTRUCTIONS[public['domain']]},
            {'role': 'user', 'content': canonical_json(public)}]


def render_prompt(tokenizer, task):
    prompt = tokenizer.apply_chat_template(messages_for(task), tokenize=False, add_generation_prompt=True)
    # AIX's R1 template opens a thinking span. Close it identically for SFT and
    # both evaluations so the trained target is the actual JSON submission.
    if prompt.rstrip().endswith('<think>'):
        prompt += '</think>\n\n'
    return prompt


def prepare_examples(*, domains, train_count, train_seed, eval_count, eval_seed, eval_difficulty, attempts=512):
    from rlvr_forge import TaskForge, solve_reference
    from verisci import verify_submission
    generation = {'seed': eval_seed, 'difficulty': eval_difficulty, 'domains': domains, 'samples_per_domain': eval_count}
    eval_cases = generate_cases(generation)
    manifest = make_manifest(eval_cases,
        {'name': 'rlvr-forge-procedural', 'revision': revision('rlvr-forge'), 'split': 'eval'},
        {'name': 'verisci-reward', 'version': revision('verisci'), 'configuration': generation}, generation)
    held_out = {problem_hash(case['input']) for case in manifest['cases']}
    if set(range(train_seed, train_seed + attempts)) & set(range(eval_seed, eval_seed + eval_count)):
        raise ValueError('training and evaluation seed ranges must be disjoint')
    forge = TaskForge(split='train')
    examples, seen, counts = [], set(), {}
    for domain in domains:
        count = 0
        for index in range(attempts):
            task = forge.generate(seed=train_seed + index, domain=domain, difficulty=1 + index % 10)
            task_dict = task.to_dict()
            digest = problem_hash(task_dict)
            if digest in held_out or digest in seen:
                continue
            submission, _ = solve_reference(task)
            if not verify_submission(task, submission).verified:
                raise ValueError('reference training answer failed verification')
            seen.add(digest)
            examples.append({'sample_id': task.task_id, 'input': public_task(task_dict),
                             'target': canonical_json(submission), 'problem_sha256': digest})
            count += 1
            if count == train_count:
                break
        if not count:
            raise ValueError(f'No distinct training problems for {domain}; choose another held-out set')
        counts[domain] = count
    audit = {'training_unique_by_domain': counts, 'requested_train_per_domain': train_count,
             'evaluation_samples': len(eval_cases), 'evaluation_unique_problems': len(held_out),
             'exact_problem_overlap': 0, 'reference_answers_verified': True}
    return examples, manifest, audit


def load_bundle(run_dir):
    root = Path(run_dir)
    config = json.loads((root / 'experiment.json').read_text())
    for name, expected in config['code_sha256'].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError('training workflow code changed; use a new experiment directory')
    manifest = json.loads((root / 'eval-manifest.json').read_text())
    validate_manifest(manifest)
    if manifest['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('evaluation manifest changed')
    if file_sha256(root / 'train.jsonl') != config['training_data_sha256']:
        raise ValueError('training data changed')
    if revision('verisci') != manifest['scorer']['version']:
        raise ValueError('VeriSci revision changed')
    examples = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    eval_hashes = {problem_hash(case['input']) for case in manifest['cases']}
    train_hashes = [problem_hash(example['input']) for example in examples]
    if not examples or len(train_hashes) != len(set(train_hashes)) or set(train_hashes) & eval_hashes:
        raise ValueError('training set is empty, duplicated or overlaps evaluation tasks')
    if checkpoint_identity(config['base_model']) != config['base_identity']:
        raise ValueError('base checkpoint changed; use a new experiment directory')
    return config, examples, manifest
