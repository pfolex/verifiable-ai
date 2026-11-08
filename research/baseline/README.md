# Clean publication baseline

`v11-local.json` was generated from the storage-baseline publication branch using the offline command in `docs/AIX-ZERO-EXPERT-STORAGE.md`. Its source hashes identify this stage of the series; later dispatch/pilot commits intentionally change some sources.

The checkpoint scored 9/105 neural-only and 105/105 with symbolic assistance. This is a 105-case template-family evaluation, not a general capability benchmark. Warm median full-response times on the measured Intel Mac were approximately 453 ms and 236 ms; hardware, settings, first-request timing, per-domain counts, and all predictions are in the report. These are observations, not latency guarantees.

The arithmetic helper can repair operands and operators and calculate answers. No checkpoint was changed. No unreviewed v12 arithmetic prototype or tokenizer additions are included. The report uses public repository-relative checkpoint identity and records actual runtime versions.

Re-run the command on your own machine; CI uploads its own measurement separately rather than pretending timings must match this Mac. An independent external reproduction is still needed before making headline capability claims.
