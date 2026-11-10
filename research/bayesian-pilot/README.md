# Reproducible Failure Capsule: Frozen-`v11` Bayesian Specialist Pilot (`BASE-14` / Issue #37)

## 1. Executive Summary

This failure capsule documents the reproducible negative result in `research/bayesian-pilot/`: training a single 49,152-parameter routed SwiGLU expert inside the frozen 817,410-parameter `AIX-Zero v11` backbone on exact rational binary Bayesian tasks (`BP` single-observation posterior and `BS` sequential conditionally-IID posterior) for 300 AdamW steps **failed its preregistered empirical promotion screen (`REJECTED`)**.

Publishing negative results as immutable, hash-verified capsules prevents survivorship bias and ensures contributors diagnose *why* frozen-backbone single-expert adaptation failed before scaling distributed training (`TRAIN-01`).

---

## 2. Capsule Artifacts & Immutable Digests

All paths in `research/bayesian-pilot/` are repository-relative (containing no private home directories):

| Artifact | Path | SHA-256 Digest / Value |
| :--- | :--- | :--- |
| **Pilot Results & Gate Decisions** | [`research/bayesian-pilot/results.json`](./results.json) | Verified by `aix_zero_tests/test_published_evidence.py` |
| **Held-Out Dataset (`BP` & `BS`)** | [`research/bayesian-pilot/dataset.json`](./dataset.json) | `218b5df4dc7f6c9a57804259df42b06f2b3e3e4909356a622014c2bd37054f30` |
| **Frozen Base Checkpoint (`v11`)** | `aix_zero/checkpoints/aix_zero_v11_kip.safetensors` | `3af801152922cd27c40c8e349bdb68b6a10efb87445716c3262b85daf5cb44cb` |
| **Rejected `BP` Candidate Shard** | [`candidates/BP/expert_bp_b6ba44e956318710feb10d1e79af1f87700fa0ac64676e198519ee461a66a44d.safetensors`](./candidates/BP/cartridge_manifest.json) | `b6ba44e956318710feb10d1e79af1f87700fa0ac64676e198519ee461a66a44d` (`49,152` params) |
| **Rejected `BS` Candidate Shard** | [`candidates/BS/expert_bs_ae43c07eca9837a4a832e9021e527640bbf8944aeb32e96a0be8c0955e5776d8.safetensors`](./candidates/BS/cartridge_manifest.json) | `ae43c07eca9837a4a832e9021e527640bbf8944aeb32e96a0be8c0955e5776d8` (`49,152` params) |

---

## 3. Learned Expert Accuracy vs. Controls vs. Exact Symbolic Solver

The capsule strictly separates **neural-only learned expert accuracy** from **non-learned controls** (`base`, `majority`, `prior_only`, `untrained_explicit_route`) and the **separate exact symbolic solver** (`solve_prompt` in [`aix_zero/bayesian_tasks.py`](../../aix_zero/bayesian_tasks.py)):

| Task | Test Prompts (`24` Held-Out `(p, a, b)` Triples) | Frozen Base (`v11`) | Majority Answer Control | Prior-Only Control (`P(H)=p`) | Learned Single-Expert Candidate (Neural-Only) | Novel-Layout Generalization (`e;b;p;a`) | Separate Exact Symbolic Solver (`solve_prompt`) | Empirical Gate Decision & Promotion Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`BP` (Single Observation)** | `96` (`48` `train_a`, `48` `novel`) | `0 / 96` (`0.00%`) | `6 / 96` (`6.25%`) | **`10 / 96` (`10.42%`)** | **`3 / 96` (`3.125%`)** | `0 / 48` (`0.00%`) | `96 / 96` (`100.0%`) | **`FAIL` (`REJECTED`)** — Below prior-only control (`-7.29%`) and `0%` novel layout |
| **`BS` (Sequential IID)** | `144` (`48` `train_a`, `48` `novel`, `48` `longer_sequence`) | `0 / 144` (`0.00%`) | `6 / 144` (`4.17%`) | **`24 / 144` (`16.67%`)** | **`1 / 144` (`0.694%`)** | `1 / 96` (`1.04%`) | `144 / 144` (`100.0%`) | **`FAIL` (`REJECTED`)** — Below prior-only control (`-15.97%`) and `1.04%` novel layout |

### Safety & Non-Regression Verification
- **Frozen Backbone Integrity:** `frozen_tensors_unchanged == True` across both runs (`frozen_tensor_sha256` verified before and after training).
- **Zero Fallback Regression:** Temporary installation and restoration of each candidate shard (`fallback_audit`) left all 105 held-out `v11` traces 100% unchanged (`changed_traces == 0`).

---

## 4. Clean-Environment Verification Command

```sh
python -m unittest aix_zero_tests/test_published_evidence.py aix_zero_tests/test_bayesian_pilot.py -v
```
