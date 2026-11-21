"""Create a deterministic portable AIX application from committed Git objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'aix.py': 'scripts/aix_release.py',
    **{path: path for path in (
        'LICENSE', 'requirements-aix-zero.txt', 'requirements-test.txt', 'artifact_io.py',
        'aix_zero/__init__.py', 'aix_zero/architecture.py', 'aix_zero/tokenizer.py',
        'aix_zero/checkpoint_registry.py', 'aix_zero/sparse_cartridge_bank.py',
        'aix_zero/checkpoints/aix_zero_v11_kip.safetensors',
    )},
}


def build(revision, output):
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}'],
        cwd=ROOT, text=True,
    ).strip()
    payloads = {
        destination: subprocess.check_output(['git', 'show', f'{revision}:{source}'], cwd=ROOT)
        for destination, source in FILES.items()
    }
    manifest = {
        'schema_version': 1, 'source_revision': revision,
        'format': 'portable-python-directory', 'checkpoint': 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors',
        'files': {path: hashlib.sha256(data).hexdigest() for path, data in sorted(payloads.items())},
    }
    payloads['build-manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    with output.open('xb') as handle:
        with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_STORED) as archive:
            for path, data in sorted(payloads.items()):
                entry = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
    return {'source_revision': revision, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.revision, args.output), indent=2))


if __name__ == '__main__':
    main()
