"""Acceptance and contract tests for #425, #412, #409, #413, #414, #416, and #417.

Clearly labels fixture-based orchestration/promotion/rollback verification separately from
neural learning outcomes (fixture success is not neural learning evidence).
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from aix_zero.autonomous_scheduler import AutonomousExperimentScheduler
from aix_zero.benchmark_baseline import load_v11
from aix_zero.champion_promotion import ChampionPromotionManager
from aix_zero.experiment_planner import (
    DEFAULT_RECIPE_CATALOG,
    persist_experiment_plan,
    plan_next_experiment,
    sanitize_manifest_for_planner,
)
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_runner import run_experiment
from aix_zero.experiment_splits import build_split_manifest, eligible_training_rows
from aix_zero.single_skill_study import (
    TRAINING_ONLY_REPLAY_POOL,
    build_single_skill_study_manifest,
    run_predeclared_training_study,
    sample_registered_replay_mixture,
)


class AutonomousImprovementLoopTests(unittest.TestCase):
    def test_training_replay_uses_registered_train_only_and_rejects_eval_leakage(self):
        base = Path(__file__).resolve().parents[1] / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer = load_v11(base)
        manifest = build_single_skill_study_manifest(tokenizer, model.config.max_seq_len)
        mix = sample_registered_replay_mixture(manifest, include_replay=True, replay_limit=2, seed=42)
        self.assertEqual(mix['target_count'], 4)
        self.assertEqual(mix['replay_count'], 2)
        self.assertEqual(mix['total_count'], 6)

        # Replay rows must never overlap development, regression, or final splits
        for split in ('development', 'regression', 'final'):
            split_rows = [r['row'] for r in manifest['payload']['splits'][split]]
            if split_rows:
                with self.assertRaises(ValueError):
                    eligible_training_rows(manifest, split_rows, lambda r: r['prompt'])

    def test_planner_consumes_failures_suppresses_repeated_recipes_and_hides_final_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'registry'
            registry = ExperimentRegistry(root)
            champion_file = Path(tmp) / 'champ.pt'
            champion_file.write_bytes(b'champion_v1')
            champ_sha = hashlib.sha256(b'champion_v1').hexdigest()

            rows = {
                s: [dict(prompt=f'{s}_q', gold=f'secret_{s}', completion='<think>1*1=1</think><ans>1</ans>', domain='META_QUANTIZATION_FP8_INT4')]
                for s in ('train', 'development', 'regression', 'final')
            }
            manifest = build_split_manifest(rows, lambda r: r['prompt'])
            safe_view = sanitize_manifest_for_planner(manifest)
            self.assertNotIn('final', safe_view['splits'])
            self.assertNotIn('secret_final', json.dumps(safe_view))
            self.assertNotIn('secret_development', json.dumps(safe_view))

            policy = dict(min_cases=1, min_net_gain=1, max_regression_losses=0, max_trace_losses=0, min_champion_regression_correct=1)
            dev_diag = dict(
                champion=dict(
                    per_domain=dict(
                        META_QUANTIZATION_FP8_INT4=dict(offered=4, answer_correct=0, trace_valid=0, failures=4),
                    )
                )
            )

            # First plan selects first catalog recipe
            p1 = plan_next_experiment(
                root,
                champion_sha256=champ_sha,
                manifest=manifest,
                policy=policy,
                development_diagnostics=dev_diag,
            )
            self.assertEqual(p1['status'], 'planned')
            self.assertEqual(p1['selected_recipe']['recipe_id'], DEFAULT_RECIPE_CATALOG[0]['recipe_id'])
            self.assertFalse(p1['used_reserved_final_split'])
            plan_path = persist_experiment_plan(root, p1)
            self.assertTrue(plan_path.is_file())

            # Record failed/rejected attempts for each recipe in the catalog and verify suppression -> idle
            for idx, rec in enumerate(DEFAULT_RECIPE_CATALOG):
                att = registry.create(
                    champion_file,
                    dict(
                        source_revision='rev',
                        tokenizer_sha256='a' * 64,
                        dataset_sha256=manifest['sha256'],
                        seed=42,
                        hardware='cpu',
                        settings=dict(
                            recipe_id=rec['recipe_id'],
                            target_domain='META_QUANTIZATION_FP8_INT4',
                            training=dict(steps=rec['steps'], learning_rate=rec['learning_rate'], max_seconds=60.0),
                        ),
                    ),
                )
                registry.transition(att, 'proposed', 'failed', 'exhausted recipe in test')
                p_next = plan_next_experiment(
                    root,
                    champion_sha256=champ_sha,
                    manifest=manifest,
                    policy=policy,
                    development_diagnostics=dev_diag,
                )
                if idx + 1 < len(DEFAULT_RECIPE_CATALOG):
                    self.assertEqual(p_next['status'], 'planned')
                    self.assertEqual(p_next['selected_recipe']['recipe_id'], DEFAULT_RECIPE_CATALOG[idx + 1]['recipe_id'])
                else:
                    self.assertEqual(p_next['status'], 'idle_all_catalog_recipes_exhausted')
                    self.assertIsNone(p_next['selected_recipe'])
                    self.assertEqual(len(p_next['suppressed_recipes']), len(DEFAULT_RECIPE_CATALOG))

    def test_atomic_promotion_cas_race_confirmation_gate_and_smoke_rollback_fixture(self):
        # Clearly labeled deterministic fixture test (#413 / #417): proves promotion/rollback mechanics,
        # not a claim of neural model improvement.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'registry'
            champ_src = Path(tmp) / 'champ.pt'
            champ_src.write_bytes(b'champion_baseline_bytes')
            mgr = ChampionPromotionManager(root)
            ptr = mgr.initialize_champion(champ_src)
            champ_sha = ptr['champion_sha256']
            self.assertEqual(ptr['version'], 1)

            rows = {
                s: [dict(prompt=f'{s}_q', gold='1', completion='1')]
                for s in ('train', 'development', 'regression', 'final')
            }
            manifest = build_split_manifest(rows, lambda r: r['prompt'])
            metadata = dict(
                source_revision='rev',
                tokenizer_sha256='c' * 64,
                dataset_sha256=manifest['sha256'],
                seed=42,
                hardware='cpu',
                settings={},
            )
            policy = dict(
                min_cases=1,
                min_net_gain=1,
                max_regression_losses=0,
                max_trace_losses=0,
                min_champion_regression_correct=1,
            )

            def make_eval(manifest_obj, a_path, b_path, split, *, dev_gain=True, reg_champ_ok=True, reg_cand_ok=True):
                a_sha = hashlib.sha256(a_path.read_bytes()).hexdigest()
                b_sha = hashlib.sha256(b_path.read_bytes()).hexdigest()
                if split == 'development':
                    a_ok, b_ok = (False, True) if dev_gain else (False, False)
                else:
                    a_ok, b_ok = reg_champ_ok, reg_cand_ok
                return dict(
                    split=split,
                    comparison_complete=True,
                    assistance_mode='neural_only',
                    settings={},
                    tokenizer_sha256='c' * 64,
                    split_manifest_sha256=manifest_obj['sha256'],
                    model_identities=dict(champion=a_sha, candidate=b_sha),
                    arms=dict(
                        champion=dict(
                            complete=True,
                            offered=1,
                            scored=1,
                            correct=int(a_ok),
                            trace_valid=int(a_ok),
                            records=[dict(prompt=f'{split}_q', gold='1', scored=True, answer_correct=a_ok, trace_valid=a_ok)],
                        ),
                        candidate=dict(
                            complete=True,
                            offered=1,
                            scored=1,
                            correct=int(b_ok),
                            trace_valid=int(b_ok),
                            records=[dict(prompt=f'{split}_q', gold='1', scored=True, answer_correct=b_ok, trace_valid=b_ok)],
                        ),
                    ),
                )

            # 1. Diagnostic eligible candidate WITHOUT independent confirmation -> retained, not promoted
            res_elig = run_experiment(
                mgr.registry,
                mgr.live_champion_path,
                manifest,
                metadata,
                policy,
                identity=lambda r: r['prompt'],
                train=lambda d, items, cfg: b'candidate_v2_bytes',
                evaluate=make_eval,
            )
            self.assertEqual(res_elig['state'], 'evaluated')
            no_conf = mgr.evaluate_and_maybe_promote(
                res_elig['attempt'],
                expected_champion_sha256=champ_sha,
                independent_confirmation=None,
            )
            self.assertFalse(no_conf['promoted'])
            self.assertEqual(mgr.current_pointer()['champion_sha256'], champ_sha)

            # 2. Stale expected_champion_sha256 (CAS race) -> raises ValueError and preserves champion
            with self.assertRaisesRegex(ValueError, 'Stale expected_champion_sha256'):
                mgr.evaluate_and_maybe_promote(
                    res_elig['attempt'],
                    expected_champion_sha256='f' * 64,
                )

            # 3. Smoke check failure -> automatic atomic rollback to exact previous champion
            valid_conf = dict(
                independent_final_test=True,
                assistance_mode='neural_only',
                champion_sha256=champ_sha,
                candidate_sha256=res_elig['candidate_sha256'],
                confirmed_improvement=True,
                evaluated_cases=10,
                net_gain=2,
                regression_losses=0,
            )
            rolled = mgr.evaluate_and_maybe_promote(
                res_elig['attempt'],
                expected_champion_sha256=champ_sha,
                independent_confirmation=valid_conf,
                smoke_check=lambda p: False,
            )
            self.assertFalse(rolled['promoted'])
            self.assertTrue(rolled['rolled_back'])
            self.assertEqual(mgr.current_pointer()['champion_sha256'], champ_sha)
            self.assertEqual(mgr.live_champion_path.read_bytes(), b'champion_baseline_bytes')
            self.assertEqual(mgr.registry.inspect(res_elig['attempt'])['state'], 'failed')

            # 4. Full passing candidate + independent confirmation + passing smoke -> atomic promotion to v2
            res_pass = run_experiment(
                mgr.registry,
                mgr.live_champion_path,
                manifest,
                metadata,
                policy,
                identity=lambda r: r['prompt'],
                train=lambda d, items, cfg: b'candidate_v3_promoted_bytes',
                evaluate=make_eval,
            )
            valid_conf_v3 = dict(valid_conf, candidate_sha256=res_pass['candidate_sha256'])
            promoted = mgr.evaluate_and_maybe_promote(
                res_pass['attempt'],
                expected_champion_sha256=champ_sha,
                independent_confirmation=valid_conf_v3,
                smoke_check=lambda p: p.read_bytes() == b'candidate_v3_promoted_bytes',
            )
            self.assertTrue(promoted['promoted'])
            self.assertEqual(mgr.current_pointer()['version'], 2)
            self.assertEqual(mgr.current_pointer()['champion_sha256'], res_pass['candidate_sha256'])
            self.assertEqual(mgr.current_pointer()['previous_champion_sha256'], champ_sha)
            self.assertEqual(mgr.registry.inspect(res_pass['attempt'])['state'], 'promoted')

    def test_scheduler_single_worker_lock_hard_timeout_and_crash_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'sched_registry'
            sched1 = AutonomousExperimentScheduler(root, max_attempt_seconds=0.3, max_total_attempts=3)
            sched2 = AutonomousExperimentScheduler(root, max_attempt_seconds=0.3, max_total_attempts=3)

            # 1. Single-worker lock blocks competing live worker
            sched1.acquire_worker_lock()
            try:
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    sched2.acquire_worker_lock()
            finally:
                sched1.release_worker_lock()

            # 2. Orphaned running attempt from crashed worker is marked failed on recovery (no duplicate promotion)
            champ = Path(tmp) / 'champ.pt'
            champ.write_bytes(b'baseline')
            manifest = build_split_manifest(
                {s: [dict(prompt=s, gold='1', completion='1')] for s in ('train', 'development', 'regression', 'final')},
                lambda r: r['prompt'],
            )
            manifest_path = Path(tmp) / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            policy_path = Path(tmp) / 'policy.json'
            policy_path.write_text(json.dumps(dict(min_cases=1, min_net_gain=1, max_regression_losses=0, max_trace_losses=0)))

            orphan = sched1.registry.create(
                champ,
                dict(
                    source_revision='rev',
                    tokenizer_sha256='a' * 64,
                    dataset_sha256=manifest['sha256'],
                    seed=1,
                    hardware='cpu',
                    settings={},
                ),
            )
            sched1.registry.transition(orphan, 'proposed', 'running', 'started before crash')
            recovered = sched1.recover_interrupted_attempts()
            self.assertEqual(recovered, [orphan])
            self.assertEqual(sched1.registry.inspect(orphan)['state'], 'failed')

            # 3. Hard subprocess deadline kills a hung worker process and records failure while preserving champion
            status = sched1.run_bounded_loop(
                champion_path=champ,
                manifest_path=manifest_path,
                policy_path=policy_path,
                max_cycles=1,
                worker_command_builder=lambda **kw: [sys.executable, '-c', 'import time; time.sleep(10)'],
            )
            self.assertEqual(status['cumulative_budget']['attempts_started'], 1)
            self.assertGreaterEqual(status['cumulative_budget']['attempts_failed'], 2)  # orphan + timeout
            self.assertEqual(status['attempt_summaries'][-1]['worker_status'], 'timeout')
            self.assertEqual(champ.read_bytes(), b'baseline')

    def test_predeclared_single_skill_study_validates_seeds_and_preserves_champion(self):
        base = Path(__file__).resolve().parents[1] / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer = load_v11(base)
        manifest = build_single_skill_study_manifest(tokenizer, model.config.max_seq_len)
        fast_recipes = (
            dict(
                recipe_id='fast_target_only',
                hypothesis='1-step smoke check without replay',
                steps=1,
                learning_rate=0.001,
                include_replay=False,
                replay_limit=0,
                trainable_scope='full_adapter_groups',
            ),
            dict(
                recipe_id='fast_with_replay',
                hypothesis='1-step smoke check with replay',
                steps=1,
                learning_rate=0.001,
                include_replay=True,
                replay_limit=1,
                trainable_scope='full_adapter_groups',
            ),
        )
        # Reject < 3 seeds or > 2 recipes
        with self.assertRaises(ValueError):
            run_predeclared_training_study(manifest=manifest, recipes=fast_recipes, seeds=(42, 43))
        with self.assertRaises(ValueError):
            run_predeclared_training_study(manifest=manifest, recipes=fast_recipes + fast_recipes, seeds=(42, 43, 44))

        report = run_predeclared_training_study(
            manifest=manifest,
            recipes=fast_recipes,
            seeds=(42, 43, 44),
            max_training_seconds=60.0,
            max_new_tokens=36,
        )
        self.assertTrue(report['champion_unchanged'])
        self.assertFalse(report['independent_final_test'])
        self.assertEqual(len(report['recipe_reports']), 2)
        for rec_rep in report['recipe_reports']:
            self.assertEqual(rec_rep['completed_seed_count'], 3)


if __name__ == '__main__':
    unittest.main()
