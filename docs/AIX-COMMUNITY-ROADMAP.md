# AIX community research roadmap

AIX aims to become a highly capable, reliable, efficient open AI system through independently verified research. World-leading capability is an aspiration, not a promised outcome or a synonym for pooled CPU count.

## What is actually known

The inspected local v11 checkpoint contains 817,410 physical parameters. Prior local evaluations measured 9/105 neural-only and 105/105 with the arithmetic helper on a narrow template-family benchmark. These are not broad frontier-model results. The helper can repair operands/operators as well as calculate arithmetic.

Local work also added integrity-checked expert snapshots, bounded caching, reversible installation, and an opt-in selected-token dispatch path. The current four-expert model did not show an end-to-end speedup with sparse dispatch, so dense remains the default. Larger 16/64-expert fixtures were synthetic performance experiments, not trained model-capability extensions.

The two frozen-base Bayesian candidates were rejected: single-observation exact answers 3/96 against a 12/96 prior-only control, and sequential answers 1/144 against a 26/144 control. Exact-solver results are separate tool-assisted results, not learned capability. The local checks reached 101 passing tests, but local work is not automatically an upstream release. BASE-01 requires review, publication, and clean-checkout reproduction before contributors depend on it.

## Success criteria

Measure capability, reliability/abstention, robustness, latency, throughput, memory, energy/cost where known, and reproducibility. Compare named systems on the same task, model/tool access, and resource budgets. Preserve uncertainty, group dependence, data rights, and contamination limitations. First aim for one independently reproduced useful gain; assess broader leadership only with current independent evidence across relevant tasks.

A larger parameter count, donated runtime, lower training loss, issue completion count, or a correct exact solver does not establish intelligence progress.

## Milestones and gates

- **M0 — Reproducible evidence and community foundations:** A clean checkout reproduces baseline evidence; claims, data provenance, and contribution rules are honest and reviewable.
- **M1 — Independently verified useful learning:** At least one specialist or adaptation recipe beats declared controls on a fresh held-out evaluation with independent reproduction; a documented no-go is valid.
- **M2 — Three-computer verified research network:** Three consenting test computers complete bounded independent jobs with verifiable results, resource enforcement, and recovery evidence.
- **M3 — Transparent public alpha:** A truthful website, working contribution route, limited enrollment, release provenance, and security readiness gates are in place.
- **M4 — Sustainable verified community throughput:** Expand only after useful throughput, review capacity, replication, privacy, and resource limits remain healthy under measured growth.
- **M5 — Independent competitive capability assessment:** Compare current appropriate reference systems under declared budgets and evidence standards; scoped leadership is assessed, never guaranteed.

Milestones have exit criteria rather than speculative delivery dates. Research tasks may finish with a well-supported negative result. Later scale studies remain blocked until learning, data, security, and throughput gates justify them.

## Workstreams

- **BASE — Reproducible foundations** (M0): Local research work must become a reviewable, reproducible public baseline before contributors build on it.
- **EVAL — Independent evaluation and measured goals** (M0): AIX must demonstrate capability gains on held-out tasks under comparable resource budgets, not optimize issue counts or one toy score.
- **DATA — Verified data and curriculum** (M0): The failed pilot requires stronger data diagnostics and generalization design before distributed workers multiply experiments.
- **TRAIN — Learning research and specialist improvement** (M1): Local Bayesian candidates scored 3/96 and 1/144, below prior-only controls. Research must diagnose that failure before scaling training.
- **REASON — Bayesian, causal, and cooperative reasoning** (M1): Version labels must correspond to precise reasoning tasks and independent verification rather than renamed arithmetic chains.
- **EXPERT — Expert library, routing, and efficient execution** (M1): Storage and selected-token dispatch exist locally; useful independently trained experts and reliable selection still require evidence.
- **WORKER — Volunteer contribution client** (M2): Begin with bounded independent jobs on trusted test computers; a public volunteer client comes after capability, isolation, and verification gates.
- **NETWORK — Coordination and distributed experiments** (M2): Independent jobs provide a tractable first network; distributed inference or synchronized training requires separate evidence.
- **SEC — Contributor safety and research integrity** (M2): Public contributions require isolation, bounded permissions, and verification of untrusted results; these are concrete product requirements.
- **WEB — Public website and transparent progress** (M3): A website should explain verified evidence, welcome contributors, and distinguish live telemetry from planned or unavailable features.
- **COMM — Community collaboration and governance** (M0): A useful community needs clear ownership, manageable review load, reproducible work, and respectful opt-in participation.
- **SCALE — Evidence-led scale and competitive capability** (M5): More computers and parameters are useful only when validated learning and useful-work efficiency justify the added complexity.

## First execution queue

1. BASE-01: review and publish the local patch series and rejected pilot evidence.
2. BASE-04: correct public claims using measured evidence.
3. DATA-01: inventory dataset/model distribution rights and provenance.
4. COMM-01 and COMM-03: establish willing reviewers and a usable contributor quickstart.
5. EVAL-01 and EVAL-02: extend the already implemented evaluation harness and define comparable success measurements.
6. TRAIN-01: analyze the rejected Bayesian experts before allocating more training compute.

Do not start all 192 tasks at once. A maintainer keeps a bounded ready queue and updates dependent issues after prerequisite acceptance. A dependency label is advisory coordination metadata; it is not automatic GitHub enforcement.

## Volunteer-compute strategy

Begin with independent, bounded evaluation jobs on three consenting computers. Resource controls, interruption handling, result verification, and useful-throughput accounting precede public enrollment. GPUs may contribute training/inference; CPUs remain useful for verification, evaluation, and small experiments. WAN-split inference and coordinated decentralized training are separate feasibility studies, not assumed scaling benefits.

## Community execution

Work is opt-in. Contributors claim ready issues with an approach, availability, and expected evidence; maintainers confirm ownership and review coverage. Contributions include tests, datasets, independent reproductions, documentation, and compute. No person is assigned without consent. Public outreach is targeted and accurate; never mass-mention researchers or describe unverified results as breakthroughs.

A research PR states its hypothesis, controls, validation budget, stopping rule, provenance, limitations, and negative results. Closing a failed hypothesis is valid when its evidence is reproducible. A release claim needs independent reproduction and a maintainer review. The backlog is revisable; prune work that evidence makes unnecessary.

## Website and progress reporting

Publish overview, contribution, roadmap, results, rejected-experiment, update, network-status, and release pages. Distinguish live telemetry from mock data; count verified useful work separately from submitted or claimed jobs. Expose aggregate approved metadata, never credentials, private worker identifiers, or final-test answers. A scheduled update cadence requires an actual owner; none is implied by this plan.

## Existing work to reuse

EVAL-004–007 are already merged (PRs #4–#7); extend their dataset manifests, paired reports, uncertainty policies, and hosted provider fixtures. Do not recreate them as new features. Related component repositories remain separate projects; tasks in this hub coordinate integration, and cross-repository implementation PRs must follow each repository's rules.

## Reference methods

- [GitHub issue/API mutation guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)
- [SWE-bench Verified and comparable evaluation settings](https://www.swebench.com/verified)
- [LiveCodeBench](https://livecodebench.github.io/)
- [Hivemind decentralized learning](https://github.com/learning-at-home/hivemind)
- [Petals distributed inference](https://petals.dev/)
- [BOINC independent volunteer-job guidelines](https://boinc.berkeley.edu/central/scientist.php)

These are references to evaluate, not claims that AIX implements their capabilities or has partnered with their communities.

Live coordination and issue index: https://github.com/pfolex/verifiable-ai/issues/10
