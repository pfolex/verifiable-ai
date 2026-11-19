"""Grade a reviewed local patch against the AIX disk-backed routing task.

Run in a sandbox appropriate for executing the submitted Python code. A fresh
checkout isolates the worktree; it does not restrict the candidate process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = '782d6be'
TARGET = 'aix_zero/sparse_cartridge_bank.py'
TASK = """Implement SparseExpertCartridgeBank.execute_routed_request(
    model, prompt, generate, *, active_base_checkpoint_sha256=None
).

Return the result of generate(model, prompt). Route approved Q:<domain>: prompts
through the matching disk-backed cartridge and force its registered expert slot.
Support two or more distinct domains with the existing catalog. Reuse the existing
temporary specialist session. The original v11 model must work unchanged after
each request. Unknown, unapproved, missing, invalid, and incompatible cartridges
use the base route. An error raised by generate must propagate and restore the
model without retrying the callback. Change only aix_zero/sparse_cartridge_bank.py.
Submit a unified git diff against the pinned base revision.
"""


def grade(patch, *, timeout=60):
    patch = patch.resolve()
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    result = dict(base=BASE, patch_sha256=digest)
    with tempfile.TemporaryDirectory(prefix='aix-repo-task-') as directory:
        checkout = Path(directory) / 'candidate'
        checkout.mkdir()
        archive = subprocess.check_output(['git', 'archive', BASE], cwd=ROOT, timeout=timeout)
        subprocess.run(['tar', '-xf', '-'], cwd=checkout, input=archive, check=True, timeout=timeout)
        stats = subprocess.run(['git', 'apply', '--numstat', '-z', str(patch)], cwd=checkout,
                               capture_output=True, text=True, timeout=timeout)
        if stats.returncode:
            return dict(result, status='invalid_patch', detail=stats.stderr.strip())
        summary = subprocess.check_output(['git', 'apply', '--summary', str(patch)],
                                          cwd=checkout, text=True, timeout=timeout)
        fields = stats.stdout.removesuffix('\x00').split('\t', 2)
        if summary or len(fields) != 3 or fields[2] != TARGET or not all(
            field.isdecimal() for field in fields[:2]
        ):
            return dict(result, status='invalid_scope')
        applied = subprocess.run(['git', 'apply', '--whitespace=error', str(patch)],
                                 cwd=checkout, capture_output=True, text=True, timeout=timeout)
        if applied.returncode:
            return dict(result, status='invalid_patch', detail=applied.stderr.strip())
        env = dict(os.environ, PYTHONPATH=str(checkout), PYTHONDONTWRITEBYTECODE='1')
        try:
            checked = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('routing_checks.py'))],
                cwd=checkout, env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return dict(result, status='timeout')
        return dict(result, status='pass' if checked.returncode == 0 else 'fail',
                    output=checked.stdout + checked.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', action='store_true', help='Print the public task and pinned revision')
    for arm in ('aix', 'peer', 'collaborative', 'reference'):
        parser.add_argument(f'--{arm}', type=Path, metavar='PATCH')
    args = parser.parse_args()
    if args.task:
        print(f'Base revision {BASE}\n\n{TASK}')
        return 0
    patches = {arm: getattr(args, arm) for arm in ('aix', 'peer', 'collaborative', 'reference')}
    if not any(patches.values()):
        parser.error('supply --task or at least one candidate patch')
    results = {arm: grade(patch) if patch else {'status': 'not_submitted'}
               for arm, patch in patches.items()}
    print(json.dumps(results, indent=2))
    return int(any(patches[arm] and row['status'] != 'pass' for arm, row in results.items()))


if __name__ == '__main__':
    sys.exit(main())
