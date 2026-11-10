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

---

## 4. Epistemic Uncertainty and Abstention Tasks (`REASON-05` / Issue #107)

To prevent models from guessing numbers when a problem is underdetermined:
- **Ground-Truth Abstention Token:** When a query omits a required prior $P(H)$, conditions on a zero-probability event, or contains an unobserved confounder $U \to X, Y$ without an instrument or backdoor set, the ground truth is `NON_IDENTIFIABLE` (or `ABSTAIN` / `INSUFFICIENT_INFO`).
- **Four-Way Outcome Separation (`score_epistemic_abstention_task` in [`aix_zero/eval_harness_bridge.py`](../aix_zero/eval_harness_bridge.py)):**
  1. `USEFUL_ABSTENTION`: Model abstains on an unidentifiable/underdetermined query (`score = 1.0`).
  2. `FALSE_CERTAINTY`: Model emits a numerical claim on an unidentifiable query (`score = 0.0`, `false_certainty = True`).
  3. `UNNECESSARY_ABSTENTION`: Model abstains on a well-posed identifiable query (`score = 0.0`).
  4. `IDENTIFIABLE_EXACT_MATCH` / `IDENTIFIABLE_WRONG_ANSWER`: Standard exact rational verification on identifiable queries.

---

## 5. Finite Cooperative-Game Tasks (`REASON-10` / Issue #95)

Cooperative transferable-utility (TU) games $(N, v)$ with players $N = \{A, B\}$ and characteristic function $v: 2^N \to \mathbb{Z}$ (`v({A}) = v_a`, `v({B}) = v_b`, `v({A,B}) = v_ab`) are verified by `score_cooperative_game_task()` in [`aix_zero/eval_harness_bridge.py`](../aix_zero/eval_harness_bridge.py) across four distinct solution concepts that must never be conflated:
1. **Grand Coalition Surplus (`grand_coalition_surplus`):** Net synergy $v(\{A, B\}) - (v(\{A\}) + v(\{B\}))$ over standalone play.
2. **Shapley Allocation for Player A (`shapley_player_a`):** Exact reduced rational $\phi_A(v) = v(\{A\}) + \frac{1}{2}\big(v(\{A,B\}) - v(\{A\}) - v(\{B\})\big)$.
3. **Shapley Allocation for Player B (`shapley_player_b`):** Exact reduced rational $\phi_B(v) = v(\{B\}) + \frac{1}{2}\big(v(\{A,B\}) - v(\{A\}) - v(\{B\})\big)$, satisfying efficiency $\phi_A(v) + \phi_B(v) = v(\{A,B\})$.
4. **Core Non-Emptiness (`core_nonempty`):** `1` if superadditive ($v(\{A,B\}) \ge v(\{A\}) + v(\{B\})$ so individually rational efficient allocations exist), else `0`.

