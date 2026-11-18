# AIX sponsorship milestones

AIX aims to help developers measure and improve AI systems through reproducible experiments. The immediate research objective is to produce verified AI engineering patches and measure whether training improves success on unseen tasks.

Progress tracker and implementation issues are maintained in [#271](https://github.com/pfolex/verifiable-ai/issues/271).

| Milestone | Deliverables | Acceptance evidence | Compute estimate |
|---|---|---|---|
| 1. Reproducible public baseline | CPU installation guide, two-cartridge demo, dated scorecard, contributor walkthrough | Clean-checkout logs, corruption rejection, restored base tensors, links to pinned reports | CPU workstation. Record elapsed time and peak memory during replication. |
| 2. Verified programming data and comparisons | Licensed repository tasks, reserved evaluation fixtures, AIX/peer/collaborative submissions | Identical grader and task budgets for all arms, raw submissions, patch success and regressions across held-out tasks | Pilot required. Model choice, token budgets, GPU-hours, and API costs remain to be measured. |
| 3. Measured learning improvement | Train a programming candidate and compare with its starting checkpoint and controls | Predeclared evaluation budget, validation-only selection, held-out results, independent reproduction | Set a spending cap after the pilot. Expand compute only when measured progress supports it. |

Milestone 1 builds on #262, #263, #264, and #267. Milestone 2 is tracked in #269 and #270. Milestone 3 depends on their results and the budget in #266. Each milestone ends with a recorded decision to continue, revise the method, or stop the experiment. Refresh this page when evidence or the budget changes, and include an update date in progress reports.

## A first contribution

Choose a bounded deliverable and post an approach on its issue before substantial implementation, following the existing contributor process.

- [#263](https://github.com/pfolex/verifiable-ai/issues/263) offers a documentation task auditing source links, settings, and dates in the scorecard. Verify each number against the linked JSON.
- [#264](https://github.com/pfolex/verifiable-ai/issues/264) offers a replication task running the installation walkthrough in a fresh supported environment and reporting exact failures and fixes.
- [#267](https://github.com/pfolex/verifiable-ai/issues/267) offers an onboarding task checking the newcomer walkthrough and submitting concrete improvements.

These are candidate starter scopes. Maintainer review capacity and assignment follow [CONTRIBUTING.md](../CONTRIBUTING.md).

## Funding and updates

The [pfolex Sponsors page](https://github.com/sponsors/pfolex) is the account funding destination. Profile activation and payout setup must be checked by the account owner before announcing a campaign.

The initial fundraising target is pending a measured pilot and dated GPU quotes. Track the budget in [#266](https://github.com/pfolex/verifiable-ai/issues/266). A budget should record GPU type, GPU-hours, storage, transfers, evaluation costs, contingency assumptions, the spending cap, and checkpoint recovery behavior. Record donated compute separately with its usage terms and contributor permission.

Publish an update after each funded experiment and monthly while funded work is active. Include the period, source/checkpoint identities, completed work, accepted and rejected results, aggregate funds received and spent, remaining funds, blockers, and the next decision. Describe outcomes using the scorecard and provide evidence links. Keep donor financial details and private account setup outside public experiment reports.
