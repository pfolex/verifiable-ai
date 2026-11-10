# Community Conduct and Moderation Rules (`COMM-02` / Issue #190)

## 1. Purpose and Scope

AIX (`verifiable-ai`) is an open experimental research project focused on verifiable, falsifiable reasoning systems. These conduct and moderation rules apply to all project spaces: GitHub issues, pull requests, discussions, code reviews, commit messages, and any official project coordination threads.

This document does **not** claim a staffed 24/7 trust-and-safety team or formal corporate ombuds office. Currently, moderation is handled by the repository owner (`@pfolex`) and any explicitly listed maintainers in [`GOVERNANCE.md`](./GOVERNANCE.md) on a best-effort basis.

---

## 2. Expected Behavior

Contributors, reviewers, and maintainers are expected to:
1. **Focus on Technical Evidence:** Base critiques on reproducible benchmarks, code, verifiers, statistical gates, and documented assumptions rather than personal attacks.
2. **Respect Consent and Boundaries:** Never assign issues, review duties, or roles to individuals without their explicit opt-in consent.
3. **Report Negative and Null Results Honestly:** Welcome falsification, negative experimental outcomes (`REJECTED` / `INCONCLUSIVE`), and bug reports without hostility or blame.
4. **Represent Project Status Accurately:** Do not post exaggerated capability claims (`"1T-ready"`, `"<25ms guaranteed"`, `"world-leading"`) or spam unrelated open-source repositories.

---

## 3. Unacceptable Behavior

The following behaviors are prohibited in AIX project spaces:
- Harassment, doxxing, personal insults, discriminatory language, or intimidation.
- Publishing private correspondence or personal data without explicit permission.
- Submitting fabricated benchmark artifacts, manipulated hashes, or undisclosed prompt/test leakage.
- Unsolicited mass-tagging (`@`-mention spamming) of external researchers or maintainers, or promotional recruitment spam in third-party communities.
- Attempting to bypass worker/artifact safety boundaries or submitting malicious code/data.

---

## 4. Reporting Conduct or Safety Concerns

- **Private Conduct or Safety Reports:** Use GitHub Private Vulnerability Reporting on `pfolex/verifiable-ai` (`Security` -> `Report a vulnerability`) or contact the repository owner (`@pfolex`) privately via their GitHub profile contact channel so sensitive details are not posted in public issues.
- **Public Technical Moderation Requests (Spam / Off-Topic / Unconsented Assignment):** Flag the comment or issue directly on GitHub and reference this policy.
- **No SLA Guarantee:** Reports are reviewed by `@pfolex` (or consenting maintainers not involved in the dispute) as capacity permits. If a report involves a maintainer, that maintainer must recuse themselves from handling it.

---

## 5. Enforcement Ladder

Maintainers may take any of the following actions proportionate to the severity and repetition of a violation:
1. **Written Warning / Comment Edit Request:** Public or private explanation of the rule violated and request to edit or retract the content.
2. **Content Removal / Thread Locking:** Hiding abusive, spammy, or off-topic comments and locking issues or pull requests to maintainers.
3. **Contribution Suspension / Rejection:** Closing pull requests or revoking opt-in coordinator/reviewer status.
4. **Repository Ban:** Blocking accounts that engage in harassment, malicious artifact submission, or repeated spam from `pfolex/verifiable-ai`.
