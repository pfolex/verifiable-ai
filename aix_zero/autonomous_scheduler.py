"""Resumable autonomous experiment scheduler with single-worker lock, subprocess boundary,
hard whole-attempt deadlines, durable phases/heartbeats, cumulative budgets, and honest
operator status/controls (#414, #416, #417).

Starts in `record_and_reject_only` mode by default; automatic live champion promotion is
enabled only when `allow_promotion=True` AND an independent confirmation protocol and
non-vacuous retention gates (#413 / #426) pass.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from aix_zero.candidate_policy import compute_retention_metrics, validate_policy
from aix_zero.champion_promotion import ChampionPromotionManager, _atomic_replace_bytes
from aix_zero.experiment_planner import (
    DEFAULT_RECIPE_CATALOG,
    persist_experiment_plan,
    plan_next_experiment,
)
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_splits import verify_manifest
from aix_zero.inspect_candidate_experiment import compare_paired_cases, inspect_attempt


def _pid_is_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class AutonomousExperimentScheduler:
    def __init__(
        self,
        registry_root,
        *,
        max_attempt_seconds=180.0,
        max_total_seconds=1800.0,
        max_total_attempts=10,
        max_total_supervised_tokens=200000,
        max_retries_per_plan=1,
        allow_promotion=False,
    ):
        self.registry = ExperimentRegistry(registry_root)
        self.root = self.registry.root
        self.lock_path = self.root / 'scheduler_worker.lock'
        self.stop_path = self.root / '.scheduler.stop'
        self.state_path = self.root / 'scheduler_state.json'
        self.max_attempt_seconds = float(max_attempt_seconds)
        self.max_total_seconds = float(max_total_seconds)
        self.max_total_attempts = int(max_total_attempts)
        self.max_total_supervised_tokens = int(max_total_supervised_tokens)
        self.max_retries_per_plan = int(max_retries_per_plan)
        self.allow_promotion = bool(allow_promotion)
        self.instance_id = uuid.uuid4().hex

    def acquire_worker_lock(self):
        """Acquire single-worker lock per registry; reclaim stale lock only if owner PID is dead."""
        payload = dict(
            pid=os.getpid(),
            instance_id=self.instance_id,
            acquired_at_utc=ExperimentRegistry._now(),
        )
        encoded = json.dumps(payload, sort_keys=True).encode()
        try:
            ExperimentRegistry._exclusive(self.lock_path, encoded)
            return payload
        except FileExistsError:
            try:
                existing = json.loads(self.lock_path.read_text())
                owner_pid = int(existing.get('pid', -1))
            except Exception:
                owner_pid = -1
            if _pid_is_alive(owner_pid):
                raise RuntimeError(f'Another autonomous scheduler worker (pid={owner_pid}) is already active on {self.root}')
            self.lock_path.unlink(missing_ok=True)
            ExperimentRegistry._exclusive(self.lock_path, encoded)
            return payload

    def release_worker_lock(self):
        if self.lock_path.exists():
            try:
                existing = json.loads(self.lock_path.read_text())
                if existing.get('instance_id') == self.instance_id or existing.get('pid') == os.getpid():
                    self.lock_path.unlink()
            except Exception:
                pass

    def request_stop(self, reason='operator_stop_requested'):
        payload = dict(stop_requested=True, reason=reason, requested_at_utc=ExperimentRegistry._now())
        _atomic_replace_bytes(self.stop_path, json.dumps(payload, indent=2, sort_keys=True).encode())
        state = self.load_state()
        state['phase'] = 'stopped'
        state['last_stop_reason'] = reason
        state['heartbeat_at_utc'] = ExperimentRegistry._now()
        self._save_state(state)
        return state

    def clear_stop(self):
        if self.stop_path.exists():
            self.stop_path.unlink()

    def should_stop(self):
        return self.stop_path.exists()

    def load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return dict(
            schema_version=1,
            phase='idle',
            mode='automatic_promotion_gated' if self.allow_promotion else 'record_and_reject_only',
            heartbeat_at_utc=ExperimentRegistry._now(),
            active_attempt=None,
            active_worker_pid=None,
            cumulative_budget=dict(
                attempts_started=0,
                attempts_completed=0,
                attempts_rejected=0,
                attempts_failed=0,
                attempts_promoted=0,
                elapsed_seconds=0.0,
                supervised_tokens=0,
                max_total_attempts=self.max_total_attempts,
                max_total_seconds=self.max_total_seconds,
                max_total_supervised_tokens=self.max_total_supervised_tokens,
            ),
            attempt_summaries=[],
            recovered_interrupted_attempts=[],
        )

    def _save_state(self, state):
        state['heartbeat_at_utc'] = ExperimentRegistry._now()
        _atomic_replace_bytes(self.state_path, json.dumps(state, indent=2, sort_keys=True, allow_nan=False).encode())

    def recover_interrupted_attempts(self):
        """Scan registry on startup and transition any orphaned `proposed` or `running` attempts to `failed`
        so crashed workers never resume implicitly or cause duplicate promotions (#406 / #414).
        """
        state = self.load_state()
        attempts_dir = self.root / 'attempts'
        recovered = []
        if attempts_dir.exists():
            for path in sorted(attempts_dir.iterdir()):
                if not path.is_dir():
                    continue
                attempt_id = path.name
                try:
                    info = self.registry.recover(attempt_id)
                except Exception:
                    continue
                if info['state'] in ('proposed', 'running'):
                    self.registry.transition(
                        attempt_id,
                        info['state'],
                        'failed',
                        f'Interrupted {info["state"]} attempt marked failed during scheduler recovery',
                    )
                    recovered.append(attempt_id)
                    state['cumulative_budget']['attempts_failed'] += 1
        if recovered:
            state['recovered_interrupted_attempts'].extend(recovered)
            state['active_attempt'] = None
            state['active_worker_pid'] = None
            state['phase'] = 'recovered_idle'
            self._save_state(state)
        return recovered

    def _execute_worker_subprocess(self, cmd):
        """Run a candidate experiment CLI in a separate process group and enforce a hard wall-clock
        deadline (`self.max_attempt_seconds`), terminating the entire process group on timeout or stop.
        """
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        state = self.load_state()
        state['active_worker_pid'] = proc.pid
        self._save_state(state)
        try:
            while proc.poll() is None:
                if self.should_stop():
                    self._terminate_process_group(proc)
                    return dict(
                        status='stopped',
                        returncode=-1,
                        elapsed_seconds=time.monotonic() - start,
                        stdout='',
                        stderr='Stopped by operator sentinel',
                    )
                if time.monotonic() - start > self.max_attempt_seconds:
                    self._terminate_process_group(proc)
                    return dict(
                        status='timeout',
                        returncode=-1,
                        elapsed_seconds=time.monotonic() - start,
                        stdout='',
                        stderr=f'Hard subprocess deadline ({self.max_attempt_seconds}s) exceeded',
                    )
                state = self.load_state()
                self._save_state(state)
                time.sleep(0.05)
            out, err = proc.communicate()
            return dict(
                status='completed' if proc.returncode == 0 else 'error',
                returncode=proc.returncode,
                elapsed_seconds=time.monotonic() - start,
                stdout=out,
                stderr=err,
            )
        finally:
            if proc.poll() is None:
                self._terminate_process_group(proc)
            state = self.load_state()
            state['active_worker_pid'] = None
            self._save_state(state)

    @staticmethod
    def _terminate_process_group(proc):
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(0.02)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    def build_operator_status(self):
        """Return an honest, inspectable operator status report (#416) linking to per-case evidence
        and distinguishing fitting, development, regression, final, and assisted results.
        Never reports 100% retention when the champion has 0 correct regression cases.
        """
        state = self.load_state()
        pointer = None
        pointer_path = self.root / 'champion_pointer.json'
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text())
        return dict(
            schema_version=1,
            registry_root=str(self.root),
            phase=state['phase'],
            mode=state['mode'],
            stop_requested=self.should_stop(),
            heartbeat_at_utc=state['heartbeat_at_utc'],
            active_worker_pid=state.get('active_worker_pid'),
            champion_pointer=pointer,
            cumulative_budget=state['cumulative_budget'],
            recovered_interrupted_attempts=state['recovered_interrupted_attempts'],
            attempt_count=len(state['attempt_summaries']),
            attempt_summaries=state['attempt_summaries'],
        )

    def run_bounded_loop(
        self,
        *,
        champion_path,
        manifest_path,
        policy_path,
        max_cycles=3,
        recipe_catalog=DEFAULT_RECIPE_CATALOG,
        worker_command_builder=None,
        independent_confirmation_fn=None,
        smoke_check_fn=None,
    ):
        """Run a bounded autonomous plan -> train -> paired evaluation -> decision -> record loop."""
        self.acquire_worker_lock()
        try:
            self.recover_interrupted_attempts()
            promotion_mgr = ChampionPromotionManager(self.root)
            pointer = promotion_mgr.initialize_champion(champion_path)
            manifest = json.loads(Path(manifest_path).read_text())
            payload = verify_manifest(manifest)
            policy = json.loads(Path(policy_path).read_text())
            validate_policy(policy)

            state = self.load_state()
            cycles_run = 0

            while cycles_run < max_cycles and not self.should_stop():
                b = state['cumulative_budget']
                if (
                    b['attempts_started'] >= self.max_total_attempts
                    or b['elapsed_seconds'] >= self.max_total_seconds
                    or b['supervised_tokens'] >= self.max_total_supervised_tokens
                ):
                    state['phase'] = 'paused_budget_exhausted'
                    self._save_state(state)
                    break

                pointer = promotion_mgr.current_pointer()
                state['phase'] = 'planning'
                self._save_state(state)

                # Build development gap diagnostics from latest attempt on current champion, or synthetic baseline gap view
                dev_rows = [r['row'] for r in payload['splits']['development']]
                latest_dev_diag = None
                for summary in reversed(state['attempt_summaries']):
                    if summary.get('champion_sha256') == pointer['champion_sha256'] and summary.get('attempt'):
                        try:
                            insp = inspect_attempt(self.root, summary['attempt'])
                            if insp.get('evaluation_present'):
                                latest_dev_diag = insp['development']
                                break
                        except Exception:
                            pass
                if latest_dev_diag is None:
                    by_dom = {}
                    for r in dev_rows:
                        dom = r.get('domain', 'unspecified')
                        entry = by_dom.setdefault(dom, dict(offered=0, answer_correct=0, trace_valid=0, failures=0))
                        entry['offered'] += 1
                        entry['failures'] += 1
                    latest_dev_diag = dict(champion=dict(per_domain=by_dom))

                plan = plan_next_experiment(
                    self.root,
                    champion_sha256=pointer['champion_sha256'],
                    manifest=manifest,
                    policy=policy,
                    development_diagnostics=latest_dev_diag,
                    recipe_catalog=recipe_catalog,
                )
                plan_path = persist_experiment_plan(self.root, plan)
                if plan['status'] != 'planned':
                    state['phase'] = plan['status']
                    state['idle_reason'] = plan['reason']
                    self._save_state(state)
                    break

                recipe = plan['selected_recipe']
                attempt_before = set(p.name for p in (self.root / 'attempts').iterdir()) if (self.root / 'attempts').exists() else set()

                state['phase'] = 'training_and_evaluating'
                state['active_plan_sha256'] = plan['plan_sha256']
                state['cumulative_budget']['attempts_started'] += 1
                self._save_state(state)

                if worker_command_builder is not None:
                    cmd = worker_command_builder(
                        champion_path=str(promotion_mgr.live_champion_path),
                        manifest_path=str(manifest_path),
                        policy_path=str(policy_path),
                        registry_path=str(self.root),
                        recipe=recipe,
                        plan=plan,
                    )
                else:
                    cmd = [
                        sys.executable,
                        '-m',
                        'aix_zero.run_candidate_experiment',
                        '--champion', str(promotion_mgr.live_champion_path),
                        '--manifest', str(manifest_path),
                        '--policy', str(policy_path),
                        '--registry', str(self.root),
                        '--steps', str(recipe['steps']),
                        '--learning-rate', str(recipe['learning_rate']),
                        '--max-training-seconds', str(recipe.get('max_seconds', 120.0)),
                        '--max-new-tokens', str(plan['evaluation_budget']['max_new_tokens']),
                        '--seed', str(plan['seed']),
                    ]

                sub_res = self._execute_worker_subprocess(cmd)
                state['cumulative_budget']['elapsed_seconds'] += sub_res['elapsed_seconds']

                attempt_after = set(p.name for p in (self.root / 'attempts').iterdir()) if (self.root / 'attempts').exists() else set()
                new_attempts = sorted(attempt_after - attempt_before)
                # Ensure any interrupted new attempt is recovered/failed cleanly
                self.recover_interrupted_attempts()
                state = self.load_state()

                if sub_res['status'] != 'completed' or not new_attempts:
                    # Record a synthetic failed attempt in registry if subprocess died before creating one
                    if not new_attempts:
                        fail_meta = dict(
                            source_revision='scheduler_worker_failure',
                            tokenizer_sha256='0' * 64,
                            dataset_sha256=manifest['sha256'],
                            seed=plan['seed'],
                            hardware='local_subprocess',
                            settings=dict(
                                recipe_id=recipe['recipe_id'],
                                target_domain=plan['selected_domain'],
                                training=dict(steps=recipe['steps'], learning_rate=recipe['learning_rate'], max_seconds=recipe.get('max_seconds', 120.0)),
                            ),
                        )
                        failed_attempt = self.registry.create(promotion_mgr.live_champion_path, fail_meta)
                        self.registry.transition(failed_attempt, 'proposed', 'failed', f"Worker subprocess {sub_res['status']}: {sub_res['stderr'][:200]}")
                        state['cumulative_budget']['attempts_failed'] += 1
                        new_attempts = [failed_attempt]

                    for att_id in new_attempts:
                        state['attempt_summaries'].append(dict(
                            attempt=att_id,
                            plan_sha256=plan['plan_sha256'],
                            plan_path=str(plan_path),
                            recipe_id=recipe['recipe_id'],
                            target_domain=plan['selected_domain'],
                            champion_sha256=pointer['champion_sha256'],
                            state='failed',
                            worker_status=sub_res['status'],
                            elapsed_seconds=round(sub_res['elapsed_seconds'], 4),
                            reason=sub_res['stderr'].strip() or sub_res['status'],
                        ))
                    self._save_state(state)
                    cycles_run += 1
                    continue

                att_id = new_attempts[-1]
                insp = inspect_attempt(self.root, att_id)
                t_metrics = insp.get('training_metrics') or {}
                sup_tokens = int(t_metrics.get('supervised_tokens_per_step', 0)) * int(t_metrics.get('steps_completed', 0))
                state['cumulative_budget']['supervised_tokens'] += sup_tokens
                state['cumulative_budget']['attempts_completed'] += 1

                promotion_result = None
                if insp['state'] == 'rejected':
                    state['cumulative_budget']['attempts_rejected'] += 1
                elif insp['state'] == 'failed':
                    state['cumulative_budget']['attempts_failed'] += 1
                elif insp['state'] == 'evaluated' and self.allow_promotion:
                    confirmation = (
                        independent_confirmation_fn(att_id, insp)
                        if independent_confirmation_fn is not None
                        else None
                    )
                    promotion_result = promotion_mgr.evaluate_and_maybe_promote(
                        att_id,
                        expected_champion_sha256=pointer['champion_sha256'],
                        independent_confirmation=confirmation,
                        smoke_check=smoke_check_fn,
                    )
                    insp = inspect_attempt(self.root, att_id)
                    if promotion_result['promoted']:
                        state['cumulative_budget']['attempts_promoted'] += 1
                    elif insp['state'] == 'rejected':
                        state['cumulative_budget']['attempts_rejected'] += 1
                    elif insp['state'] == 'failed':
                        state['cumulative_budget']['attempts_failed'] += 1

                ret = insp.get('retention') or {}
                dev = insp.get('development') or {}
                reg = insp.get('regression') or {}
                state['attempt_summaries'].append(dict(
                    attempt=att_id,
                    plan_sha256=plan['plan_sha256'],
                    plan_path=str(plan_path),
                    recipe_id=recipe['recipe_id'],
                    target_domain=plan['selected_domain'],
                    champion_sha256=insp['champion_sha256'],
                    candidate_sha256=insp.get('candidate_sha256'),
                    dataset_sha256=insp['dataset_sha256'],
                    state=insp['state'],
                    decision=insp.get('verified_decision', {}).get('decision'),
                    reasons=insp.get('verified_decision', {}).get('reasons', []),
                    training_loss_before=t_metrics.get('train_loss_before'),
                    training_loss_after=t_metrics.get('train_loss_after'),
                    supervised_tokens=sup_tokens,
                    elapsed_seconds=round(sub_res['elapsed_seconds'], 4),
                    development_results=dict(
                        assistance_mode='neural_only',
                        champion_correct=dev.get('champion', {}).get('answer_correct'),
                        candidate_correct=dev.get('candidate', {}).get('answer_correct'),
                        offered=dev.get('champion', {}).get('offered'),
                        net_gain=insp.get('verified_decision', {}).get('development_net_gain'),
                    ),
                    regression_retention_results=dict(
                        assistance_mode='neural_only',
                        champion_correct=ret.get('champion_correct'),
                        candidate_correct=ret.get('candidate_correct'),
                        retained_champion_correct=ret.get('retained_champion_correct'),
                        retention_denominator=ret.get('retention_denominator'),
                        retention_measurable=ret.get('retention_measurable'),
                        retention_rate=ret.get('retention_rate'),
                        retention_status=ret.get('retention_status'),
                        losses=ret.get('losses'),
                    ),
                    final_test_status='NOT_EVALUATED' if not (promotion_result and promotion_result.get('promoted')) else 'CONFIRMED_INDEPENDENT',
                    promotion_result=promotion_result,
                    evaluation_artifact_path=str(self.root / 'attempts' / att_id / 'evaluation.json'),
                ))
                state['phase'] = 'idle_between_attempts'
                self._save_state(state)
                cycles_run += 1

            return self.build_operator_status()
        finally:
            self.release_worker_lock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('status', help='Print read-only operator status and per-attempt evidence summaries')
    p_stop = sub.add_parser('stop', help='Write scheduler stop sentinel')
    p_stop.add_argument('--reason', default='operator_stop_command')
    sub.add_parser('recover', help='Recover interrupted attempts without resuming training')

    p_run = sub.add_parser('run', help='Run one-shot or bounded multi-attempt autonomous loop')
    p_run.add_argument('--champion', type=Path, required=True)
    p_run.add_argument('--manifest', type=Path, required=True)
    p_run.add_argument('--policy', type=Path, required=True)
    p_run.add_argument('--max-cycles', type=int, default=1)
    p_run.add_argument('--max-attempt-seconds', type=float, default=180.0)
    p_run.add_argument('--max-total-seconds', type=float, default=1800.0)
    p_run.add_argument('--max-total-attempts', type=int, default=10)
    p_run.add_argument('--allow-promotion', action='store_true')

    args = parser.parse_args(argv)
    scheduler = AutonomousExperimentScheduler(
        args.registry,
        max_attempt_seconds=getattr(args, 'max_attempt_seconds', 180.0),
        max_total_seconds=getattr(args, 'max_total_seconds', 1800.0),
        max_total_attempts=getattr(args, 'max_total_attempts', 10),
        allow_promotion=getattr(args, 'allow_promotion', False),
    )

    if args.command == 'status':
        out = scheduler.build_operator_status()
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
        return out
    if args.command == 'stop':
        out = scheduler.request_stop(reason=args.reason)
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
        return out
    if args.command == 'recover':
        recovered = scheduler.recover_interrupted_attempts()
        out = dict(recovered_attempts=recovered, status=scheduler.build_operator_status())
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
        return out
    if args.command == 'run':
        scheduler.clear_stop()
        out = scheduler.run_bounded_loop(
            champion_path=args.champion,
            manifest_path=args.manifest,
            policy_path=args.policy,
            max_cycles=args.max_cycles,
        )
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
        return out


if __name__ == '__main__':
    main()
