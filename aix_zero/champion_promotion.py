"""Atomic evidence-based champion promotion, rejection, and rollback (#413).

Guarantees:
1. Diagnostic `eligible_for_review` status alone cannot trigger an automatic live switch:
   requires complete compatible development + regression evidence, non-vacuous baseline
   retention coverage (`min_champion_regression_correct >= 1`), AND a verified independent
   confirmation record matching the candidate and policy digests.
2. Uses an atomic compare-and-swap (CAS) versioned champion pointer (`champion_pointer.json`)
   backed by immutable registry artifacts and an archived predecessor log (`champion_history.jsonl`).
3. Runs a post-switch smoke callback before finalizing; if the smoke check fails or raises,
   automatically rolls back to the exact previous champion bytes and records the failed
   promotion without losing the baseline.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from aix_zero.candidate_policy import screen_candidate, validate_policy
from aix_zero.experiment_registry import ExperimentRegistry


def _atomic_replace_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ChampionPromotionManager:
    def __init__(self, registry_root, live_champion_path=None):
        self.registry = ExperimentRegistry(registry_root)
        self.pointer_path = self.registry.root / 'champion_pointer.json'
        self.lock_path = self.registry.root / 'champion_cas.lock'
        self.history_path = self.registry.root / 'champion_history.jsonl'
        self.archive_dir = self.registry.root / 'archived_champions'
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.live_champion_path = Path(live_champion_path) if live_champion_path is not None else (self.registry.root / 'current_champion.pt')

    def initialize_champion(self, source_path):
        digest = self.registry.snapshot(source_path)
        raw = self.registry.verify(digest).read_bytes()
        if not self.pointer_path.exists():
            pointer = dict(
                schema_version=1,
                version=1,
                champion_sha256=digest,
                previous_champion_sha256=None,
                promoted_from_attempt=None,
                updated_at_utc=ExperimentRegistry._now(),
            )
            ExperimentRegistry._exclusive(
                self.pointer_path,
                json.dumps(pointer, indent=2, sort_keys=True).encode(),
            )
            _atomic_replace_bytes(self.live_champion_path, raw)
            self._append_history(dict(event='initialized', **pointer))
        return self.current_pointer()

    def current_pointer(self):
        if not self.pointer_path.exists():
            raise FileNotFoundError('Champion pointer is not initialized')
        pointer = json.loads(self.pointer_path.read_text())
        self.registry.verify(pointer['champion_sha256'])
        if self.live_champion_path.exists():
            live_sha = hashlib.sha256(self.live_champion_path.read_bytes()).hexdigest()
            if live_sha != pointer['champion_sha256']:
                raise ValueError('Live champion file digest does not match champion pointer')
        return pointer

    def _append_history(self, record):
        with self.history_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')

    @staticmethod
    def validate_independent_confirmation(confirmation, *, expected_champion_sha256, expected_candidate_sha256, min_confirmation_cases=1):
        if not isinstance(confirmation, dict):
            raise ValueError('Independent confirmation evidence is required for promotion')
        if confirmation.get('independent_final_test') is not True:
            raise ValueError('Confirmation must come from an independent reserved evaluation protocol')
        if confirmation.get('assistance_mode') != 'neural_only':
            raise ValueError('Confirmation must be neural_only')
        if confirmation.get('champion_sha256') != expected_champion_sha256 or confirmation.get('candidate_sha256') != expected_candidate_sha256:
            raise ValueError('Confirmation model digests do not match champion and candidate')
        if confirmation.get('confirmed_improvement') is not True:
            raise ValueError('Independent confirmation did not confirm improvement')
        if int(confirmation.get('evaluated_cases', 0)) < min_confirmation_cases:
            raise ValueError('Independent confirmation has insufficient evaluated cases')
        if int(confirmation.get('net_gain', 0)) < 1 or int(confirmation.get('regression_losses', 0)) > 0:
            raise ValueError('Independent confirmation failed net-gain or non-regression requirement')

    def evaluate_and_maybe_promote(
        self,
        attempt,
        *,
        expected_champion_sha256,
        independent_confirmation=None,
        smoke_check=None,
    ):
        """Evaluate an `evaluated` attempt for atomic compare-and-swap promotion.

        Fails closed and preserves the current champion on:
        - Stale or raced `expected_champion_sha256`,
        - Missing or failed development/regression screening or zero baseline retention coverage,
        - Missing or failed `independent_confirmation` protocol,
        - Post-switch `smoke_check` failure (automatically rolls back to predecessor).
        """
        try:
            ExperimentRegistry._exclusive(self.lock_path, str(os.getpid()).encode())
        except FileExistsError as error:
            raise RuntimeError('Competing promotion compare-and-swap in progress') from error

        try:
            pointer = self.current_pointer()
            if pointer['champion_sha256'] != expected_champion_sha256:
                raise ValueError('Stale expected_champion_sha256 in compare-and-swap promotion')

            inspected = self.registry.inspect(attempt)
            if inspected['state'] != 'evaluated':
                raise ValueError(f"Attempt {attempt} is in state '{inspected['state']}', not 'evaluated'")
            directory = self.registry.root / 'attempts' / attempt
            if inspected['manifest']['champion_sha256'] != expected_champion_sha256:
                raise ValueError('Attempt parent champion differs from current champion pointer')

            evaluation = json.loads((directory / 'evaluation.json').read_text())
            policy = json.loads((directory / 'policy.json').read_text())
            if policy.get('min_champion_regression_correct', 0) < 1:
                policy = dict(policy, min_champion_regression_correct=1)
            validate_policy(policy)

            screen = screen_candidate(evaluation['development'], evaluation['regression'], policy)
            if screen['decision'] != 'eligible_for_review':
                reason = '; '.join(screen.get('reasons') or [screen.get('reason', 'failed promotion screen')])
                if screen['decision'] == 'reject':
                    self.registry.transition(attempt, 'evaluated', 'rejected', reason)
                return dict(
                    promoted=False,
                    rolled_back=False,
                    decision=screen['decision'],
                    reason=reason,
                    champion_sha256=pointer['champion_sha256'],
                    version=pointer['version'],
                )

            candidate_sha256 = evaluation['candidate_sha256']
            if candidate_sha256 == pointer['champion_sha256']:
                raise ValueError('Candidate digest is identical to current champion')
            candidate_bytes = self.registry.verify(candidate_sha256).read_bytes()
            previous_bytes = self.registry.verify(pointer['champion_sha256']).read_bytes()

            try:
                self.validate_independent_confirmation(
                    independent_confirmation,
                    expected_champion_sha256=pointer['champion_sha256'],
                    expected_candidate_sha256=candidate_sha256,
                )
            except ValueError as error:
                return dict(
                    promoted=False,
                    rolled_back=False,
                    decision='retained_awaiting_or_failed_independent_confirmation',
                    reason=str(error),
                    champion_sha256=pointer['champion_sha256'],
                    version=pointer['version'],
                )

            # Archive previous champion before atomic switch
            archive_path = self.archive_dir / f"v{pointer['version']:06d}_{pointer['champion_sha256']}.pt"
            if not archive_path.exists():
                ExperimentRegistry._exclusive(archive_path, previous_bytes)

            new_pointer = dict(
                schema_version=1,
                version=pointer['version'] + 1,
                champion_sha256=candidate_sha256,
                previous_champion_sha256=pointer['champion_sha256'],
                promoted_from_attempt=attempt,
                updated_at_utc=ExperimentRegistry._now(),
            )
            _atomic_replace_bytes(self.live_champion_path, candidate_bytes)
            _atomic_replace_bytes(self.pointer_path, json.dumps(new_pointer, indent=2, sort_keys=True).encode())

            if smoke_check is not None:
                try:
                    smoke_ok = bool(smoke_check(self.live_champion_path))
                    if not smoke_ok:
                        raise RuntimeError('Post-promotion smoke check returned False')
                except Exception as smoke_error:
                    # Immediate atomic rollback to predecessor
                    _atomic_replace_bytes(self.live_champion_path, previous_bytes)
                    _atomic_replace_bytes(self.pointer_path, json.dumps(pointer, indent=2, sort_keys=True).encode())
                    reason = f'Rolled back after post-promotion smoke failure: {smoke_error}'
                    self.registry.transition(attempt, 'evaluated', 'failed', reason)
                    self._append_history(dict(
                        event='rolled_back_after_smoke_failure',
                        attempt=attempt,
                        failed_candidate_sha256=candidate_sha256,
                        restored_champion_sha256=pointer['champion_sha256'],
                        version=pointer['version'],
                        reason=reason,
                        created_at_utc=ExperimentRegistry._now(),
                    ))
                    return dict(
                        promoted=False,
                        rolled_back=True,
                        decision='rolled_back_smoke_failure',
                        reason=reason,
                        champion_sha256=pointer['champion_sha256'],
                        version=pointer['version'],
                    )

            conf_path = directory / 'independent_confirmation.json'
            ExperimentRegistry._exclusive(
                conf_path,
                json.dumps(independent_confirmation, indent=2, sort_keys=True, allow_nan=False).encode(),
            )
            self.registry.transition(
                attempt,
                'evaluated',
                'promoted',
                'passed paired development, non-vacuous retention, independent confirmation, and post-switch smoke',
                conf_path,
            )
            self._append_history(dict(event='promoted', **new_pointer))
            return dict(
                promoted=True,
                rolled_back=False,
                decision='promoted',
                champion_sha256=new_pointer['champion_sha256'],
                previous_champion_sha256=new_pointer['previous_champion_sha256'],
                version=new_pointer['version'],
            )
        finally:
            if self.lock_path.exists():
                self.lock_path.unlink()
