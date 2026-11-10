# Contributing to AIX

AIX is an experimental open research project. Current results and limitations are described in the community roadmap; world-leading AI is an aspiration, not a contributor promise.

## Find work

Use the roadmap issue and `aix:roadmap` plus `status:ready` labels. Begin with a small documentation, test-fixture, data-verification, or reproduction task. `status:blocked` means prerequisite acceptance is still needed. Stable task codes such as BASE-01 identify dependencies even when issue numbers differ.

Comment on a ready issue with the scope you propose, your available time, and how you will demonstrate completion. Wait for a maintainer to confirm ownership/review coverage before substantial work. Do not claim multiple large tasks or assign other people without their agreement. No response-time guarantee is implied.

## Set up

Start from the current main branch and read the issue's prerequisites. The local AIX-Zero research patch series must first be reviewed and published under BASE-01. Do not assume every proposed file already exists upstream. Reuse existing requirements-test.txt and requirements-training.txt instructions as appropriate; the reviewed minimal research environment is tracked by BASE-02. Ordinary contributions should be testable offline without paid credentials.

## Make a reviewable contribution

Use a focused branch and PR linked to one issue. Describe the concrete behavior, why it matters, what changed, and meaningful validation. Preserve unrelated work. Include relevant tests and limitations. Keep model/data artifacts out of ordinary Git changes unless the repository's artifact policy explicitly permits them.

For experiments, include hypothesis, controls, trainable parameter budget, source/model/data hashes, seeds, resource budget, checkpoint-selection rule, validation/final-test separation, and results including failures. Compare strongest declared shortcut controls. Separate model-only results, arithmetic repair, exact solvers, retrieval, and external models. A lower loss or a larger parameter count is not a capability result.

Dataset contributions need redistribution rights, provenance, split identity, generator/verifier code, and negative checks. Do not submit private or sensitive information. Compute contributors run only approved workloads with explicit consent and enforced resource limits; public enrollment is not available until its readiness gate passes.

## Review and completion

A maintainer verifies acceptance criteria and reproduction evidence before closing work. Research can conclude with rejection or inconclusive evidence; document why. Independent reproduction is required before headline capability claims. Code authors do not self-approve promotion of trained artifacts. The first reviewer/maintainer roles are opt-in vacancies until people explicitly accept them.

## Communication

Use the community coordination issue for interests and review availability, and task comments for technical coordination. Be respectful and evidence-focused. Do not mass-mention people, promise compensation, imply affiliations, or post promotional claims in unrelated repositories. External recruitment messages need an actual maintainer-selected destination and accurate project status.

Live coordination and issue index: https://github.com/pfolex/verifiable-ai/issues/10
