"""Failure-driven skill selection and bounded experiment planner (#409).

Consumes development failure diagnostics and the experiment registry ledger to choose
at most one bounded hypothesis/recipe from an explicit recipe catalog while:
1. Suppressing repeated failed or rejected recipes for the same champion and skill gap,
2. Never reading reserved `final` split prompts or answers,
3. Persisting the frozen plan rationale, budgets, seeds, and success criteria before training,
4. Idling with an explicit reason when all bounded catalog recipes have already been exhausted.
"""
from copy import deepcopy
import hashlib
import json

from aix_zero.candidate_policy import validate_policy
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_splits import verify_manifest
from aix_zero.inspect_candidate_experiment import inspect_attempt


DEFAULT_RECIPE_CATALOG = (
    dict(
        recipe_id='replay_balanced_lr1e3_steps10',
        action_category='replay_and_data_balance',
        hypothesis='Mix registered training-only replay anchors with target-skill training examples to prevent catastrophic forgetting while training target domain.',
        steps=10,
        learning_rate=0.001,
        max_seconds=120.0,
        include_replay=True,
        replay_limit=2,
        trainable_scope='full_adapter_groups',
    ),
    dict(
        recipe_id='conservative_lr4e4_steps15',
        action_category='optimizer_and_step_schedule',
        hypothesis='Use a lower learning rate (4e-4) with 15 bounded steps and replay anchors to reduce representation drift on shared backbone layers.',
        steps=15,
        learning_rate=0.0004,
        max_seconds=120.0,
        include_replay=True,
        replay_limit=2,
        trainable_scope='full_adapter_groups',
    ),
    dict(
        recipe_id='single_expert_slot_lr1e3_steps15',
        action_category='parameter_scope_isolation',
        hypothesis='Restrict gradient updates to a single routed expert slot so shared router/decoder weights remain unchanged.',
        steps=15,
        learning_rate=0.001,
        max_seconds=120.0,
        include_replay=False,
        replay_limit=0,
        trainable_scope='single_routed_expert',
    ),
)


def sanitize_manifest_for_planner(manifest):
    """Return a verified copy of `manifest` with `final` split records stripped so reserved
    final-evaluation prompts or answers can never enter planning (#409).
    """
    payload = verify_manifest(manifest)
    return dict(
        schema_version=payload['schema_version'],
        dataset_sha256=manifest['sha256'],
        splits=dict(
            train=deepcopy(payload['splits']['train']),
            development=[
                dict(identity=r['identity'], family=r['family'],
                     row=dict(prompt=r['row']['prompt'], domain=r['row'].get('domain', 'unspecified')))
                for r in payload['splits']['development']
            ],
            regression=[
                dict(identity=r['identity'], family=r['family'],
                     row=dict(prompt=r['row']['prompt'], domain=r['row'].get('domain', 'unspecified')))
                for r in payload['splits']['regression']
            ],
        ),
    )


def summarize_skill_gaps(development_diagnostics):
    """Aggregate development failures by domain into ranked measurable skill gaps."""
    per_domain = development_diagnostics['champion']['per_domain']
    gaps = []
    for domain, stats in sorted(per_domain.items()):
        offered = stats['offered']
        correct = stats['answer_correct']
        trace_ok = stats['trace_valid']
        failures = offered - correct
        trace_failures = offered - trace_ok
        if failures > 0 or trace_failures > 0:
            gaps.append(dict(
                domain=domain,
                offered=offered,
                champion_answer_correct=correct,
                champion_trace_valid=trace_ok,
                answer_failures=failures,
                trace_failures=trace_failures,
                failure_rate=round(failures / max(1, offered), 6),
            ))
    gaps.sort(key=lambda g: (-g['answer_failures'], -g['trace_failures'], g['domain']))
    return gaps


def plan_next_experiment(
    registry_root,
    *,
    champion_sha256,
    manifest,
    policy,
    development_diagnostics,
    recipe_catalog=DEFAULT_RECIPE_CATALOG,
    seed=42,
    max_new_tokens=36,
):
    """Select one bounded hypothesis/recipe for the highest-priority development skill gap,
    suppressing recipes that already failed or were rejected for `(champion_sha256, target_domain)`.
    """
    if not isinstance(champion_sha256, str) or len(champion_sha256) != 64:
        raise ValueError('Valid 64-char champion_sha256 is required')
    policy_sha256 = validate_policy(policy)
    safe_view = sanitize_manifest_for_planner(manifest)
    if 'final' in safe_view['splits']:
        raise RuntimeError('Reserved final split must not be present in planner view')

    gaps = summarize_skill_gaps(development_diagnostics)
    if not gaps:
        plan = dict(
            schema_version=1,
            status='idle_no_skill_gaps',
            reason='Champion has zero answer or trace failures on development diagnostics',
            champion_sha256=champion_sha256,
            dataset_sha256=safe_view['dataset_sha256'],
            policy_sha256=policy_sha256,
            selected_domain=None,
            selected_recipe=None,
            suppressed_recipes=[],
            used_reserved_final_split=False,
        )
        plan['plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        return plan

    target_gap = gaps[0]
    target_domain = target_gap['domain']

    # Scan registry history for previously failed/rejected attempts on the same champion
    registry = ExperimentRegistry(registry_root)
    attempts_dir = registry.root / 'attempts'
    exhausted_recipe_ids = set()
    suppressed = []
    history_summary = []

    if attempts_dir.exists():
        for attempt_path in sorted(attempts_dir.iterdir()):
            if not attempt_path.is_dir():
                continue
            try:
                info = registry.inspect(attempt_path.name)
            except Exception:
                continue
            manifest_meta = info['manifest']
            if manifest_meta.get('champion_sha256') != champion_sha256:
                continue
            state = info['state']
            settings = manifest_meta['metadata'].get('settings', {})
            rec_id = settings.get('recipe_id')
            rec_domain = settings.get('target_domain', target_domain)
            training_cfg = settings.get('training', {})
            history_summary.append(dict(
                attempt=attempt_path.name,
                state=state,
                recipe_id=rec_id,
                target_domain=rec_domain,
                steps=training_cfg.get('steps'),
                learning_rate=training_cfg.get('learning_rate'),
            ))
            if state in ('rejected', 'failed') and rec_domain == target_domain:
                if rec_id:
                    exhausted_recipe_ids.add(rec_id)
                    suppressed.append(dict(recipe_id=rec_id, attempt=attempt_path.name, state=state))
                else:
                    # Also match by (steps, learning_rate, include_replay, trainable_scope) signature
                    for candidate_rec in recipe_catalog:
                        if (
                            candidate_rec['steps'] == training_cfg.get('steps')
                            and candidate_rec['learning_rate'] == training_cfg.get('learning_rate')
                        ):
                            exhausted_recipe_ids.add(candidate_rec['recipe_id'])
                            suppressed.append(dict(
                                recipe_id=candidate_rec['recipe_id'],
                                attempt=attempt_path.name,
                                state=state,
                            ))

    selected_recipe = None
    for candidate_rec in recipe_catalog:
        if candidate_rec['recipe_id'] not in exhausted_recipe_ids:
            selected_recipe = deepcopy(candidate_rec)
            break

    if selected_recipe is None:
        plan = dict(
            schema_version=1,
            status='idle_all_catalog_recipes_exhausted',
            reason=f'All {len(recipe_catalog)} bounded catalog recipes have already failed or been rejected for champion {champion_sha256[:12]} on {target_domain}',
            champion_sha256=champion_sha256,
            dataset_sha256=safe_view['dataset_sha256'],
            policy=deepcopy(policy),
            policy_sha256=policy_sha256,
            skill_gaps=gaps,
            selected_domain=target_domain,
            selected_recipe=None,
            suppressed_recipes=suppressed,
            used_reserved_final_split=False,
        )
        plan['plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        return plan

    plan = dict(
        schema_version=1,
        status='planned',
        reason=f'Selected {selected_recipe["recipe_id"]} targeting {target_domain} ({target_gap["answer_failures"]}/{target_gap["offered"]} development failures)',
        champion_sha256=champion_sha256,
        dataset_sha256=safe_view['dataset_sha256'],
        policy=deepcopy(policy),
        policy_sha256=policy_sha256,
        seed=int(seed),
        evaluation_budget=dict(
            max_new_tokens=int(max_new_tokens),
            num_candidates=1,
            temperature=0.0,
            inflight_alu=False,
        ),
        skill_gaps=gaps,
        selected_domain=target_domain,
        selected_recipe=selected_recipe,
        suppressed_recipes=suppressed,
        used_reserved_final_split=False,
    )
    plan['plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def persist_experiment_plan(registry_root, plan):
    """Persist a frozen experiment plan into `<registry>/plans/<plan_sha256>.json` before training."""
    registry = ExperimentRegistry(registry_root)
    plans_dir = registry.root / 'plans'
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_sha = plan['plan_sha256']
    dest = plans_dir / f'{plan_sha}.json'
    data = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False).encode()
    try:
        ExperimentRegistry._exclusive(dest, data)
    except FileExistsError:
        pass
    if hashlib.sha256(dest.read_bytes()).hexdigest() != hashlib.sha256(data).hexdigest():
        raise ValueError('Persisted plan digest mismatch')
    return dest
