# Contributing to AIX (`COMM-03` / Issue #69)

AIX (`verifiable-ai`) is an experimental open research project building verifiable, falsifiable neuro-symbolic reasoning systems. Current baseline results (`817,410` physical parameters in `v11`; `9/105` neural-only vs. `105/105` symbolic-assisted on held-out template arithmetic) and limitations are documented in the [Community Research Roadmap (Issue #10)](https://github.com/pfolex/verifiable-ai/issues/10). World-leading AI is an aspiration, not a contributor promise.

All participants must follow [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md), the opt-in governance rules in [`GOVERNANCE.md`](./GOVERNANCE.md), and the Apache-2.0 software licensing and provenance rules in [`LICENSE`](./LICENSE) and [`docs/LICENSE-AND-PROVENANCE-POLICY.md`](./docs/LICENSE-AND-PROVENANCE-POLICY.md).

---

## 1. Zero-Credential Offline Quickstart (Under 5 Minutes on CPU)

Every core test and baseline verification in AIX runs **100% offline on CPU** without paid API keys, hosted LLM credentials, or GPUs.

```sh
# 1. Clone the repository and create a Python 3.11+ virtual environment
git clone https://github.com/pfolex/verifiable-ai.git
cd verifiable-ai
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-aix-zero.txt

# 2. Clone the pinned BranchBox dependency required by curriculum modules
git clone https://github.com/pfolex/branchbox.git branchbox
git -C branchbox checkout 49873a0662ad13de7943619bf8fd822d3d3458a9

# 3. Run the evaluation harness test suite (80+ tests) and AIX-Zero test suite (30+ tests)
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
.venv/bin/python -m unittest discover -s aix_zero_tests -p "test_*.py" -v
```

---

## 2. Five Concrete Contributor Paths

Choose the path that matches your skills and available time:

### Path A: Code & Architecture (`ARCH-*`, `BASE-*`)
- **What you do:** Improve model components (`aix_zero/architecture.py`), sparse expert dispatch (`MixtureOfReasoningExperts`), cartridge storage (`aix_zero/sparse_cartridge_bank.py`), or distributed worker infrastructure (`fishtest.py`).
- **Verification command:**
  ```sh
  .venv/bin/python -m unittest discover -s aix_zero_tests -p "test_*.py" -v
  ```
- **Guardrail:** Never overwrite `aix_zero/checkpoints/aix_zero_v11_kip.safetensors` or committed `research/` JSON evidence files on `main`.

### Path B: Evaluation, Verifiers & Statistical Gates (`EVAL-*`)
- **What you do:** Build deterministic domain verifiers, baseline controls, leakage audits, or statistical promotion gates using [`evaluation_result.py`](./evaluation_result.py), [`benchmark_runner.py`](./benchmark_runner.py), [`statistical_gate.py`](./statistical_gate.py), and [`aix_zero/eval_harness_bridge.py`](./aix_zero/eval_harness_bridge.py).
- **Verification command:**
  ```sh
  .venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
  .venv/bin/python -m unittest aix_zero_tests/test_eval_harness_bridge.py -v
  ```
- **Guardrail:** Always report `neural_only`, `symbolic_assisted`, and exact-solver results separately.

### Path C: Datasets & Provenance (`DATA-*`)
- **What you do:** Author procedural task generators with exact ground-truth verifiers, audit train/validation/test split disjointness, and register dataset provenance in [`research/dataset-provenance-catalog.json`](./research/dataset-provenance-catalog.json) validated by [`aix_zero/data_provenance.py`](./aix_zero/data_provenance.py).
- **Verification command:**
  ```sh
  .venv/bin/python -m unittest aix_zero_tests/test_data_provenance.py -v
  ```
- **Guardrail:** Any dataset with `restricted` or `unknown` license status, personal data, or unverified teacher output is automatically blocked from public volunteer jobs.

### Path D: Independent Reproduction & Documentation (`COMM-*`)
- **What you do:** Reproduce published baselines on your own CPU hardware, audit documentation for scientific accuracy, or improve contributor guides.
- **Verification command (offline v11 baseline reproduction):**
  ```sh
  .venv/bin/python -m aix_zero.benchmark_baseline \
    --output /tmp/my-v11-reproduction.json
  ```
- **How to report:** Post the generated JSON summary (`neural_only` score, `symbolic_assisted` score, hardware model, and SHA-256 digests) on the relevant issue.

### Path E: Opt-In Compute Contributions (`DIST-*`)
- **What you do:** Run bounded verification or specialist training tasks once the public enrollment readiness gate (`DIST-11`) passes.
- **Current status:** Public volunteer compute enrollment is **not yet open** until `DIST-01` through `DIST-11` safety, quota, and sandboxing gates are completed. Local single-machine testing is available via `python -m unittest tests/test_fishtest.py`.

---

## 3. Finding and Claiming Work (`COMM-04` / Issue #100)

1. Browse open issues labeled `aix:roadmap` and `status:ready` (see the master index in [Issue #10](https://github.com/pfolex/verifiable-ai/issues/10)).
2. `status:blocked` means prerequisite tasks (identified by stable codes like `BASE-01`, `EVAL-01`, `DATA-01`) must be completed and accepted first. Do not begin substantial implementation on a `status:blocked` task unless your work is strictly to unblock its prerequisite.
3. **One Active Owner + Optional Collaborators:** To claim a `status:ready` issue, post a structured public comment using the template below. Only one active primary owner should coordinate a task at a time; additional contributors may join as optional collaborators with the active owner's agreement.
   ```markdown
   ### Claim Proposal (`<TASK-CODE>`)
   - **Active Owner:** @<your-github-handle>
   - **Optional Collaborators:** (none or @handle with explicit opt-in)
   - **Prerequisites Verified:** `<list closed prerequisite issue numbers and commit SHAs>`
   - **Blockers / Open Questions:** `<none or specific technical blocker>`
   - **Proposed Approach:** `<1-3 sentences describing files/modules to add or update>`
   - **Closing Evidence Plan:** `<exact test commands, artifacts, and negative checks that will prove completion>`
   ```
4. **No Unconsented Assignment or Review Promises:** Never assign an issue to another person, list someone as a collaborator, or promise reviewer availability without their explicit written agreement. Maintainers confirm review coverage as capacity permits; no response-time SLA is implied.
5. **Releasing a Claim:** If your availability changes, post a short comment releasing the issue so another contributor can pick it up.

---

## 4. Submitting a Reviewable Pull Request

- Use a focused branch and PR linked to one issue (`Fixes #<number>`).
- For experiments, include:
  - Hypothesis, controls (majority, prior-only, base checkpoint), and trainable parameter budget,
  - Source, model, tokenizer, and dataset SHA-256 digests,
  - Validation vs. held-out test separation and checkpoint-selection rule,
  - Full results including negative or inconclusive outcomes (`REJECTED` / `INCONCLUSIVE`).
- Keep large binary checkpoints or unverified artifacts out of ordinary Git commits unless explicitly required by the issue's artifact policy.

Live coordination and issue index: https://github.com/pfolex/verifiable-ai/issues/10
