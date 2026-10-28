#!/usr/bin/env zsh
# Create and push all 6 standalone repositories in verifiable-ai to GitHub (user: pfolex)
set -euo pipefail

GH_BIN="${HOME}/.local/bin/gh"
GITHUB_USER="${1:-pfolex}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

typeset -A DESCRIPTIONS
DESCRIPTIONS=(
  verisci     "Universal ground-truth scientific & formal verification cartridges (Logic, Silicon, Systems, Matter)"
  rlvr-forge  "Seed-locked procedural task synthesis & RLVR/GRPO environments across scientific domains"
  branchbox   "Deterministic Copy-on-Write execution DAG sandbox for AI agents"
  mlx-tree    "Apple Silicon prefix-shared Radix KV-cache & local GRPO LoRA parameter delta trainer"
  steerlens   "Top-K Sparse Autoencoders (SAEs) & parameter-folded Contrastive Activation Addition steering"
  everbench   "Contamination-proof benchmark, Stockfish SPRT statistical gate & TIES/DARE parameter merger"
)

for repo in verisci rlvr-forge branchbox mlx-tree steerlens everbench; do
  desc="${DESCRIPTIONS[$repo]}"
  echo "==> Creating & pushing ${GITHUB_USER}/${repo} ..."
  cd "${ROOT_DIR}/${repo}"
  git branch -M main

  if ! "${GH_BIN}" repo view "${GITHUB_USER}/${repo}" >/dev/null 2>&1; then
    "${GH_BIN}" repo create "${GITHUB_USER}/${repo}" --public --description "${desc}" --source=. --remote=origin --push
  else
    git remote set-url origin "git@github.com:${GITHUB_USER}/${repo}.git" 2>/dev/null || \
      git remote add origin "git@github.com:${GITHUB_USER}/${repo}.git"
    git push -u origin main
  fi
done

echo "All 6 repositories created and pushed to https://github.com/${GITHUB_USER}!"
