"""Cooperative cycle-boundary pause. POSIX advisory locks are never deleted."""
from contextlib import contextmanager
from functools import wraps
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

PAUSE = '.checkpoint.pause.json'
CYCLE_LOCK = '.checkpoint-cycle.lock'
PUBLISH_LOCK = '.checkpoint-publisher.lock'
_local = threading.local()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


@contextmanager
def file_lock(path, timeout=0):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink(): raise ValueError('Lock must not be a symlink')
    with path.open('a+') as stream:
        end = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= end: raise TimeoutError('Checkpoint lock timed out: '+path.name)
                time.sleep(.05)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def source_fingerprint(repo):
    repo = Path(repo)
    paths = sorted(set(repo.glob('*.py')) | set((repo/'aix_zero').glob('*.py')))
    entries = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def cycle_boundary(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        state = self.state_dir.resolve(); key = str(state)
        held = getattr(_local, 'held', set())
        if key in held: return function(self, *args, **kwargs)
        with file_lock(state/CYCLE_LOCK):
            if (state/PAUSE).exists():
                status = self.get_status()
                status.update(daemon_state='PAUSED_FOR_CHECKPOINT', checkpoint_cycle_phase='paused',
                              checkpoint_guard_pid=os.getpid(), checkpoint_protocol_version=1)
                atomic_json(self.status_path, status)
                return dict(executed=False, halted_by_kill_switch=False, daemon_state='PAUSED_FOR_CHECKPOINT')
            if not hasattr(self, '_checkpoint_source_fingerprint'):
                self._checkpoint_source_fingerprint = source_fingerprint(getattr(self, 'repo_root', Path(__file__).resolve().parents[1]))
            status = self.get_status()
            status.update(checkpoint_cycle_phase='active', checkpoint_guard_pid=os.getpid(), checkpoint_protocol_version=1,
                          checkpoint_runtime_source_sha256=self._checkpoint_source_fingerprint)
            if function.__name__ == 'run_single_self_improvement_cycle':
                bank = kwargs.get('cartridge_bank_dir') or state/'meta_ai_cartridges'
                status['checkpoint_cartridge_bank_dir'] = str(Path(bank).resolve())
            atomic_json(self.status_path, status)
            try:
                _local.held = held | {key}
                result = function(self, *args, **kwargs)
                latest = self.get_status()
                phase = 'idle'
                if function.__name__ == 'run_single_self_improvement_cycle' and (not result.get('executed') or self.should_stop()):
                    phase = 'interrupted'
                latest.update(checkpoint_cycle_phase=phase, checkpoint_guard_pid=os.getpid(), checkpoint_protocol_version=1,
                              checkpoint_runtime_source_sha256=self._checkpoint_source_fingerprint)
                if 'checkpoint_cartridge_bank_dir' in status:
                    latest['checkpoint_cartridge_bank_dir'] = status['checkpoint_cartridge_bank_dir']
                atomic_json(self.status_path, latest)
                return result
            except BaseException:
                status = self.get_status(); status['checkpoint_cycle_phase'] = 'failed'
                atomic_json(self.status_path, status)
                raise
            finally:
                _local.held = held
    return wrapped
