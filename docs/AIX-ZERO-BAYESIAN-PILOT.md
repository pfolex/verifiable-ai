# Frozen-base Bayesian specialist pilot

This pilot trains two independent 49,152-parameter routed expert candidates from
v11 while freezing every other model tensor. It does not expand the vocabulary,
train the verifier/router/shared expert, overwrite the base checkpoint, or activate
candidates in normal queries.

## Tasks and reference calculations

The machine-readable prompts define a binary hypothesis H with prior p=P(H),
a=P(E=1|H), and b=P(E=1|not H). Example:

```text
Q:BP:p=1/2;a=3/4;b=1/4;e=1;P=
```

The correct posterior is 3/4. A BS prompt uses a binary observation string instead
of a single observation. For p=1/2, a=3/4, b=1/4, observing 11 yields 9/10.
Sequential observations are assumed conditionally independent and identically
distributed given H; this is an explicit model assumption, not something inferred
from arbitrary data.

Ground truth enumerates the finite joint probability distribution using exact
fractions. A separate assisted solver applies sequential Bayes updates. Tests and
dataset construction check agreement. The existing in-flight integer ALU is
not used because floor division and operand repair are inappropriate here.

Neural outputs are scored as exact rational probabilities inside answer tags;
equivalent fractions count as correct. Decimal strings are not accepted. There is
no rounding tolerance, hidden arithmetic repair, or test-time answer injection in
the neural-only condition. Assistance means running the full exact task solver:
its accuracy is reported separately and does not demonstrate learned reasoning.
This pilot measures point posterior answers, not calibration of model confidence
or broader Bayesian epistemology.

## Splits and controls

Each of p, a, b is drawn from {1/5, 1/3, 1/2, 2/3, 4/5}. The 125 parameter triples
are deterministically split with seed 42 into 85 training, 16 validation, and 24
held-out test groups. All observations/layouts of a triple stay in one split.
The two candidates share the same split assignments and train independently.

Training uses two field orders. Validation uses unseen parameter triples in a
training order. Test sets include additional unseen triples in both a known order
and a held-out field order. Sequential tests also include three-observation
sequences, whereas training uses only two observations. These paired cases are
not independent statistical trials. Posterior answer values can recur across
splits; this is parameter and layout generalization, not answer-value exclusion.

Controls are the unchanged v11 model, the same untrained expert under the explicit
route, the most common training answer, and a predictor that returns the prior
without reading the evidence. The exact solver is an assisted reference, not a
control that the neural candidate is expected to beat.

## Training and selection

Run from the repository root with the existing Python 3.11 environment:

```sh
.venv-training/bin/python -m aix_zero.train_bayesian_pilot \
  --output-dir training-runs/bayesian-pilot --steps 300
.venv-training/bin/python -m unittest discover -s aix_zero_tests -v
```

The run directory must not exist. Default budget: 300 updates per task, batch 16,
AdamW learning rate .001 and weight decay .01, gradient clipping at 1, and one CPU
thread. BP trains original slot 0; BS independently trains original slot 1 from
the unchanged v11 checkpoint. Each request explicitly selects that one routed
expert plus the frozen shared expert. The untrained-route control isolates the
change of routing policy from the effect of learned weights.

Only answer-completion tokens and EOS contribute to training loss; prompt and
padding labels are ignored. Validation completion loss is checked every 60 steps
and at the end, and selects the checkpoint (including the initial checkpoint if
training does not improve validation loss). Test outcomes do not select weights.
Evaluation is greedy, one candidate, maximum 24 generated tokens, with no symbolic
helper. Original-domain restoration checks use a 40-token cap and both neural-only
and existing integer-ALU-assisted modes.

Hashes verify that every non-expert tensor stays unchanged during training, the
serving base is restored after temporary installation, and the original checkpoint
file remains unchanged. Per-task artifacts contain the selected expert, versioned
manifest, full hashes, training configuration and logs, all evaluation outputs,
controls, and the empirical promotion decision. Model and dataset/source hashes
are recorded in the run-level report.

## Promotion screen and use

The fixed engineering screen requires at least 24 held-out parameter groups,
accuracy at least 10 percentage points above the best neural/constant/prior-only
control, at least 25% accuracy on the held-out layouts (including longer sequences
for BS), and no changes to existing-domain fallback traces after restoring the base.
This is an empirical screen, not SPRT, a confidence bound, or a significance claim.
The manifest's statistical SPRT field remains NOT_EVALUATED.

Candidates are kept in separate directories. A failed candidate is REJECTED; a
passing candidate is ELIGIBLE_NOT_ACTIVATED. Neither is loaded into normal REPL
queries by this command. A candidate's measured accuracy applies to the explicit
single-expert route, not merely to installing it while leaving the learned router
in control.

For controlled inspection, the `temporary_specialist(model, bank, task)` context
in `aix_zero.train_bayesian_pilot` installs the candidate, selects its explicit
route, then restores its slot and route even on request failure. The caller must
hold exclusive model access. It is an experiment helper, not a concurrent server
or a bypass of promotion decisions. An ordinary unrelated request runs on the
untouched base outside this context.

The fallback audit loads a candidate from disk, executes a request, restores the
base, then compares all 105 original-domain traces against a freshly loaded v11
reference in neural-only and integer-ALU-assisted modes. This demonstrates the
restoration policy; it does not claim that forcing a Bayesian expert onto unrelated
tasks avoids forgetting.

## Published evidence capsule

`research/bayesian-pilot/` contains a clean publication-branch reproduction: dataset,
selected rejected `.safetensors` shards, manifests, validation logs, all task outputs,
and original-domain restoration audits. Model and candidate identities are portable
repository/run-relative paths; local home directories are not published.

The routing policy, checkpoint/data/source identities, and evaluation budget are
recorded in the manifests. Measured specialist accuracy applies to the explicit
route, not an ordinary router-controlled installation. These final-test examples
are now public and consumed; future curriculum tuning must use validation data and
reserve fresh final tests. This is a local reproduction, not an independent external
review. No rejected candidate is activated in normal queries.
