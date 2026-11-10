# Final Roadmap Scaling Decision: Scale, Redirect, or Stop (`SCALE-16` / Issue #215)

## Executive Decision

- **Decision:** `REDIRECT_TO_JOINT_BACKBONE_AND_VERIFIED_TRACE_RESEARCH_BEFORE_VOLUNTEER_SCALING`
- **Large-Scale Public Volunteer Compute:** `STOPPED / LOCKED` until a multi-layer or joint backbone+router adaptation recipe demonstrates an independently reproduced held-out exact-match gain over all shortcut controls (`TRAIN-16`).
- **Programmatic Gate:** [`publish_final_roadmap_scaling_decision()`](../aix_zero/eval_harness_bridge.py)

## Evidence Synthesis Across All Milestones (`M0` – `M6`)

1. **What Succeeded & Is Reproducibly Verified:**
   - **Deterministic Verifiable Evaluation (`M0`–`M1`):** Content-addressed checkpoints (`v11` `817,410` physical parameters, `719,106` active per-token under top-2 routing, `3` recurrent passes), semantic-group disjoint splits (`BP`/`BS` `v1.0.0`), and exact rational/causal/cooperative reference verifiers (`EVAL-01`..`EVAL-16`, `REASON-01`..`REASON-16`).
   - **Symbolic-Assisted vs. Neural-Only Separation:** `AIX-Zero v11` achieves `105/105 (100.0%)` with in-flight symbolic arithmetic repair (`inflight_alu=True`) and `9/105 (8.57%)` neural-only (`inflight_alu=False`).
   - **Safe Cartridge & Coordinator Mechanics (`M3`–`M5`):** Atomic `.safetensors` expert snapshots, temporary session rollback, idempotent lease queues, and `2x` replication guards are fully unit-tested.
2. **What Failed & Why We Redirect Before Scaling (`M2`):**
   - Training a single routed expert (`32,768` parameters) inside the frozen `v11` backbone reduced training token loss (`9.146 -> 0.564` on `BP`; `9.373 -> 0.506` on `BS`) but achieved `0/96 (0.0%)` exact match on held-out semantic groups, losing to the `prior_only` control (`12/96 = 12.5%`).
   - Therefore, scaling out single-expert frozen-backbone jobs across volunteer machines would waste contributor electricity without producing generalizable reasoning.

## Next Bounded Hypothesis

- **Hypothesis ID:** `H-V12-JOINT-LATENT-CORE-AND-STEP-VERIFIER`
- **Scope:** Unfreeze the shared recurrent latent core + `lm_head` alongside routed expert slots under reference-verified intermediate trace supervision (`REASON-14`) on a local single-GPU/CPU budget, requiring `>= +10%` held-out exact-match improvement over `prior_only` and `majority_class` controls before any distributed scaling is reconsidered.
- **Honesty Commitment:** Completing the 192-issue engineering and verification backlog establishes a rigorous, auditable research harness—it does **not** promise inevitable world-leading intelligence (`promises_inevitable_world_leading_intelligence: false`).
