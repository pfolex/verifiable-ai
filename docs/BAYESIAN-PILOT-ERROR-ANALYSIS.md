# Error Analysis of the Rejected Bayesian Single-Expert Pilot (`TRAIN-01` / Issue #76)

This analysis diagnoses why the single-expert frozen-backbone Bayesian pilot (`research/bayesian-pilot/results.json`) scored **`3 / 96` (`3.125%`)** on `BP` and **`1 / 144` (`0.694%`)** on `BS` (`REJECTED`), underperforming the trivial `prior_only` control (`10 / 96` = `10.42%` on `BP`, `24 / 144` = `16.67%` on `BS`).

---

## 1. Quantitative Failure Breakdown

| Task | Held-Out Test Prompts | Learned Candidate (`neural_only`) | Total Errors | `train_a` Layout (`p;a;b;e`) | `novel` Layout (`e;b;p;a`) | `longer_sequence` (`len(e)=3`) | `prior_only` Shortcut Control | Majority Answer Control |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BP` (Single Observation)** | `96` | `3 / 96` (`3.125%`) | **`93`** | `3 / 48` (`6.25%`) | `0 / 48` (`0.00%`) | N/A | `10 / 96` (`10.42%`) | `6 / 96` (`6.25%`) |
| **`BS` (Sequential IID)** | `144` | `1 / 144` (`0.694%`) | **`143`** | `0 / 48` (`0.00%`) | `1 / 48` (`2.08%`) | `0 / 48` (`0.00%`) | `24 / 144` (`16.67%`) | `6 / 144` (`4.17%`) |

---

## 2. Root-Cause Failure Categories

1. **Frozen Encoder, Router, Shared Expert, and Pointer-Head Bottleneck:**
   - Only `49,152` parameters (one routed SwiGLU FFN in `latent_reasoning_core`) were trainable while the `768,258` backbone parameters—including the encoder attention, token embeddings, prompt-pointer head, and decoder—remained frozen to `v11` weights.
   - Because `v11`'s prompt-pointer head (`pointer_temperature = 0.35`) strongly biases output logits toward copying tokens that appear in the input prompt (such as the unupdated prior `p` or likelihood denominators), a single frozen-backbone FFN expert could not override the frozen decoder/pointer readout to emit novel reduced denominators $p \cdot a + (1-p) \cdot b$.
2. **Below-Shortcut Performance (`learned < prior_only`):**
   - Whenever the true posterior happened to equal the prior ($a = b$), `prior_only` scored `100%`, giving `10/96` on `BP` and `24/144` on `BS`. The trained expert perturbed logits away from copying `p` without learning exact rational multiplication and GCD reduction, resulting in worse-than-prior accuracy (`-7.29%` on `BP`, `-15.97%` on `BS`).
3. **Complete Layout & Length Generalization Collapse (`0.00%`–`2.08%`):**
   - Even on validation (`12.5%` on `BP`, `4.17%` on `BS`), the learned expert relied on positional token offsets of `p=...;a=...;b=...;e=...`. Reordering fields (`e;b;p;a`) collapsed `BP` accuracy to `0/48` (`0.00%`), and extending `BS` to 3 observations collapsed accuracy to `0/48` (`0.00%`).

---

## 3. Protocol Guardrails for Subsequent Training (`TRAIN-02+`)

- **No Test-Set Hyperparameter Selection:** The consumed `BP`/`BS` test split (`dataset.json`) is never used to tune learning rates, step counts, or trainable parameter scopes.
- **Validation-Only Curriculum Partition (`DATA-08`):** All future representation or multi-module adaptation experiments must be tuned exclusively on the validation partition via `build_validation_curriculum_experiment_set()` and audited against `audit_answer_frequency_shortcuts()` (`DATA-07`).
