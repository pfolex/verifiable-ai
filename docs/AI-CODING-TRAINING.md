# First AI-coding training experiment

This opt-in CPU experiment teaches short Python expressions for scalar SGD,
linear layers and LoRA scaling. It is a prerequisite exercise, not an AI coding
expert, PyTorch training-loop implementation, autonomous patcher, or external
coding benchmark. The daemon does not invoke this trainer.

Run from the repository with its Python environment. Stop the daemon first and
use a preserved, compatible champion checkpoint; do not read a checkpoint while
another process may be writing it.

```sh
python -m aix_zero.train_ai_coding \
  --champion ./candidate-inputs/champion.pt \
  --output ./coding-experiments/first-run \
  --steps 20 --learning-rate 0.0001 --max-seconds 120 --seed 42
```

The output directory must not already exist. The checkpoint loader supports the
existing v11 backbone and physical-LoRA candidate formats; it rejects incompatible
architectures. All model parameters are trained for this pilot, including base
weights if adapters exist. The champion is read only. Outputs are candidate.pt
and report.json; nothing is promoted, published, or loaded into the daemon.

The 24 training examples use coefficients 1–8; the 12 development examples use
9–12 with shared templates. This measures coefficient generalization within a
very narrow curriculum. There is no independent final test or retention suite.
Do not promote this candidate without separate retention and broader coding
checks. A lower training loss alone does not demonstrate coding progress.

Generation is greedy, neural-only, at most 32 new tokens, without symbolic answer
repair, peer assistance or cartridge routing. The functional scorer interprets
only arithmetic AST nodes and named numeric inputs; it never evals or executes
generated Python. Eight input assignments detect common sign, scaling and
constant-output errors, but finite probes cannot prove arbitrary equivalence.

The time limit is cooperative between optimizer steps and excludes evaluation;
it is not a hard process deadline. The report records before/after generated
answers, checkpoint hashes, physical parameter count, loss history and settings.

Next stages should introduce distinct task templates, untouched final tests,
retention gates, and actual function/patch generation tested in an isolated
execution environment before connecting this to the autonomous scheduler.
