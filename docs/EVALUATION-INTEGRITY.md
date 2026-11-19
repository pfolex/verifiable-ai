# Interpreting autopilot evaluation results

The arena schema 2.0.0 measures generated neural answers and retains each prompt, completion, reference answer, and score. Generation receives the prompt and fixed settings. Reference answers are used afterward for scoring. A separate symbolic control records the prompt-based solver's accuracy.

Run the arena with the normal control CLI from the repository root.

```sh
python -m aix_zero.aix_ctl arena --cycle 1
```

The default loads v11 and, when present, the evolving checkpoint from the chosen state directory. Incompatible evolving weights raise an error. Autopilot passes its in-memory candidate directly. Each report records the tensor-state digest, tokenizer digest, generation settings, total tasks, and per-domain counts. Historical scorecards remain historical artifacts.

The probes use a distinct cycle index from the current training curriculum. They share task templates and have no guarantee of disjoint semantic identities or answers. The field `independent_final_test` is false. Use a reserved problem-family evaluation for promotion or capability claims.

The legacy field `reflected_heldout_exact_match` records symbolic repair assistance and carries `reflection_assistance_mode: symbolic_solver`. A prompt-only repair helper builds the candidate before the compatibility wrapper scores it. Unknown prompts retain the original response, so their reference answer cannot be copied into a repair. This metric does not establish learned self-correction.

SWE patch counts identify `predefined_reference_fixtures`. Actual model-written patch performance belongs in the three-arm comparison tracked by #270. An arena accuracy of 100% cannot establish a sequential statistical result. The field `sprt_non_regression_status` stays `NOT_EVALUATED` until an appropriate comparison is implemented.

Interrupted reports set `evaluation_complete` false. A report with no completed tasks has null accuracy. Real generation makes arena runs slower than the earlier solver-only checks; the number of evaluated tasks is recorded explicitly. Changes are tracked in #369.
