"""Check the minimal AIX environment without importing the training stack."""
from importlib import metadata
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    print(f'Python {sys.version.split()[0]}')
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        failures.append('Use Python 3.11 or 3.12 for the pinned research dependencies.')
    for package, expected in (('torch', '2.2.2'), ('numpy', '1.26.4'), ('safetensors', '0.8.0')):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            failures.append(f'Missing {package}. Install requirements-aix-zero.txt with this Python.')
            continue
        print(f'{package} {version}')
        if version.split('+')[0] != expected:
            failures.append(f'{package} requires {expected} for the baseline environment.')
    try:
        version = metadata.version('jsonschema')
        if not (4, 18) <= tuple(map(int, version.split('.')[:2])) < (5, 0):
            failures.append('jsonschema requires >=4.18,<5.')
    except metadata.PackageNotFoundError:
        failures.append('Missing jsonschema. Install requirements-aix-zero.txt.')
    checkpoint = ROOT / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
    if checkpoint.is_file():
        print(f'v11 checkpoint SHA-256 {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}')
    else:
        failures.append('Missing committed v11 checkpoint. Restore it from the repository.')
    for failure in failures:
        print(f'ERROR {failure}', file=sys.stderr)
    if failures:
        return 1
    print('Minimal environment ready. Run python -m aix_zero.demo_cartridges from the repository root.')
    if not (ROOT / 'branchbox/src/branchbox').is_dir():
        print('For the full baseline and test suite, install BranchBox as shown in docs/GETTING-STARTED.md.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
