# Reasoning Capability Specification (`REASON-01`)

This specification defines the target reasoning capabilities, formal assumptions, and evaluation boundaries for **AIX-Zero** across Bayesian, Causal, and Game-Theoretic reasoning (`REASON-01`, Issue `#73`).

---

## 1. Core Distinction: Toy Exact Inference vs. Open-Ended Reasoning

Every reasoning claim in `verifiable-ai` must distinguish between:
1. **Closed-World Toy Exact Inference:** Structured problems with explicit parameters where an exact symbolic solver (e.g., Bayes' rule, finite-table $do$-calculus, finite bimatrix payoff dominance/Nash solver) computes a unique ground-truth answer.
2. **Ambiguous / Underdetermined Scenarios:** Problems where critical priors, causal graphs, or payoff matrices are missing or unidentifiable, and the correct output is an explicit **non-identifiability or abstention statement** rather than a hallucinated number.
3. **Open-Ended Calibration:** Probabilistic forecasts where accuracy is evaluated via proper scoring rules (`Brier score`, `log loss`, and `Expected Calibration Error` via `evaluate_probability_calibration()` in [`aix_zero/eval_harness_bridge.py`](../aix_zero/eval_harness_bridge.py)).

---

## 2. Capability Matrix by Track

| Track | Domain Codes | Formal Task Definition | Required Assumptions | Target Metric & Evaluation Boundary |
| :--- | :--- | :--- | :--- | :--- |
| **Bayesian Updating (`REASON-02`)** | `BP`, `BS` | Compute posterior $P(H \mid E) = \frac{P(E \mid H)P(H)}{P(E)}$ over binary or finite hypotheses, including multi-step sequential evidence updates and base-rate neglect traps. | Explicit prior $P(H) \in (0,1)$, likelihoods $P(E \mid H)$ and $P(E \mid \neg H)$, and conditional independence of sequential observations when stated. | Exact reduced fraction (`p/q`) or decimal within $\pm 10^{-4}$ via `score_rational_or_probability()`, plus Brier score & ECE calibration. |
| **Causal Inference (`REASON-03`)** | `CAUSAL_DO` (planned extension beyond `v11`) | Distinguish observational conditioning $P(Y \mid X=x)$ from interventional queries $P(Y \mid do(X=x))$ and counterfactual queries on finite DAGs with observed confounders $Z$ (backdoor adjustment) or unobserved confounders (non-identifiable). | Explicit DAG structure or structural equations over finite discrete variables; positivity $P(X=x \mid Z=z) > 0$. | Exact interventional probability when identifiable; explicit `NON_IDENTIFIABLE` classification when an unobserved confounder blocks identification. |
| **Strategic / Game-Theoretic (`REASON-04`)** | `GAME_2X2` (planned extension beyond `v11`) | Identify strictly dominant strategies, pure/mixed Nash equilibria, Pareto-optimal outcomes, and backward-induction subgame equilibria in finite 2-player games. | Complete finite payoff matrix or extensive-form tree with common knowledge of rationality. | Exact strategy profile / equilibrium payoff verification against symbolic game solver; explicit multiplicity reporting when multiple equilibria exist. |

---

## 3. Baseline Status in `v11` (`817,410` Learned Parameters)

- **Included in `v11` 21-Domain Curriculum:**
  - `BP` (`Bayesian Post`): `0/5` (`0.0%`) neural-only, `5/5` (`100.0%`) with symbolic verification/repair.
  - `BS` (`Bayes Seq`): `0/5` (`0.0%`) neural-only, `5/5` (`100.0%`) with symbolic verification/repair.
- **Not Yet in `v11` Checkpoint:**
  - Dedicated `CAUSAL_DO` and `GAME_2X2` evaluation suites are specified here as target expansion tracks (`REASON-03`, `REASON-04`) and must not be claimed as capabilities of the frozen `v11` checkpoint (`aix_zero_v11_kip.safetensors`).
