# Website Public-Alpha Readiness Audit (`WEB-16` / Issue #189)

## Audit Summary

- **Public-Alpha Documentation & Data Builders:** `READY_WITH_HONEST_DISCLOSURES`
- **Programmatic Gate:** [`audit_website_public_alpha_readiness()`](../aix_zero/website_data_builder.py)

## Verification Checklist

1. **Onboarding & Issue Discoverability:**
   - New contributors can locate `status:ready` issues via [`docs/COMMUNITY-TEMPLATES-AND-ONBOARDING.md`](./COMMUNITY-TEMPLATES-AND-ONBOARDING.md) and run a self-contained local evaluation via [`docs/CONTRIBUTOR-WORKER-ONBOARDING.md`](./CONTRIBUTOR-WORKER-ONBOARDING.md).
2. **Strict Separation of Modes & Negative Results:**
   - [`build_reproducible_results_page_payload()`](../aix_zero/website_data_builder.py) separates `v11` unassisted neural-only accuracy (`9/105 = 8.57%`) from symbolic ALU-assisted accuracy (`105/105 = 100.0%`).
   - [`build_rejected_experiment_archive_payload()`](../aix_zero/website_data_builder.py) publishes the `REJECTED` Bayesian single-expert pilot (`0/96 = 0.0%` held-out accuracy) and states that lower training loss does not imply reasoning.
3. **Enrollment & Frontier Disclaimers:**
   - [`build_contributor_installation_and_consent_page()`](../aix_zero/website_data_builder.py) keeps public volunteer enrollment disabled until `SEC-16` and `TRAIN-16` pass, and no public page promises frontier performance.
