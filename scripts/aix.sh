#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_env="${AIX_TRAINING_ENV:-${repo_root}/.venv-training}"
if [[ "${1:-}" == "setup" ]]; then
  if command -v uv >/dev/null 2>&1; then
    if [[ ! -x "$training_env/bin/python" ]]; then
      uv venv --python 3.12 "$training_env"
    fi
    uv pip install --python "$training_env/bin/python" -r "$repo_root/requirements-training.txt"
  else
    if [[ ! -x "$training_env/bin/python" ]]; then
      python3 -m venv "$training_env"
    fi
    "$training_env/bin/python" -m pip install --upgrade pip
    "$training_env/bin/python" -m pip install -r "$repo_root/requirements-training.txt"
  fi
  printf 'Environment ready. Run: ./scripts/aix.sh doctor\n'
  exit 0
fi
if [[ ! -x "$training_env/bin/python" ]]; then
  printf 'Run ./scripts/aix.sh setup first.\n' >&2
  exit 2
fi
exec "$training_env/bin/python" "$repo_root/train_aix.py" "$@"
