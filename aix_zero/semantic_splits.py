"""Semantic problem identity canonicalization and disjoint split verification (`DATA-02` / Issue #43).

Ensures that train, validation, and test splits have strictly empty intersections under
canonical semantic identity across:
1. Binary Bayesian probability triples (`p`, `a`, `b` reduced exact fractions, regardless of
   prompt field ordering `p;a;b;e` vs `e;b;p;a` or unreduced fraction formatting `2/4` vs `1/2`),
2. Algebraic & arithmetic templates (`ALGEBRA_SOLVE`, `ARITH_CHAIN`, `SYSTEM_2X2`, `CALCULUS_DERIV`,
   `CODE_TRACE`, etc., normalizing whitespace and variable/field formatting),
3. Causal intervention / observation graphs (`CAUSAL_DO`),
4. Normal-form strategic games (`SYMBIOTIC_GAME`).
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from artifact_io import canonical_json


def _normalize_fraction_str(raw: str) -> str:
    f = Fraction(raw.strip())
    return f"{f.numerator}/{f.denominator}"


def canonical_problem_identity(item: Dict[str, Any] | str) -> Dict[str, Any]:
    """Compute a formatting-invariant semantic problem identity object for a task or prompt."""
    if isinstance(item, str):
        prompt = item.strip()
        domain = None
    elif isinstance(item, dict):
        if "input" in item and isinstance(item["input"], dict):
            inner = item["input"]
            prompt = str(inner.get("prompt", "")).strip()
            domain = inner.get("domain")
            if "spec" in inner:
                return {
                    "kind": "verisci_spec",
                    "domain": str(domain),
                    "spec": inner["spec"],
                }
        else:
            prompt = str(item.get("prompt", "")).strip()
            domain = item.get("domain") or item.get("task")
    else:
        raise TypeError("Expected a task dict or prompt string")

    # Collapse whitespace for formatting invariance
    compact = re.sub(r"\s+", "", prompt)

    # 1. Bayesian tasks (Q:BP:...;P= or Q:BS:...;P=)
    # Group identity is the ordered triple of exact reduced probabilities (p, a, b),
    # invariant to layout order (p;a;b;e vs a;p;b;e vs e;b;p;a) and observation string e.
    m_bayes = re.fullmatch(r"Q:(BP|BS):([a-zA-Z0-9/=;]+);P=", compact)
    if m_bayes:
        _, body = m_bayes.groups()
        kv = {}
        for part in body.split(";"):
            k, sep, v = part.partition("=")
            if not sep:
                raise ValueError(f"Malformed Bayesian field in {prompt!r}")
            kv[k] = v
        if {"p", "a", "b"}.issubset(kv):
            return {
                "kind": "bayesian_triple",
                "p": _normalize_fraction_str(kv["p"]),
                "a": _normalize_fraction_str(kv["a"]),
                "b": _normalize_fraction_str(kv["b"]),
            }

    # 2. Causal graph / intervention tasks (Q:Causal:... or domain == CAUSAL_DO)
    if compact.startswith("Q:Causal:") or domain == "CAUSAL_DO":
        body = compact.split("Q:Causal:", 1)[-1]
        assignments = sorted(
            f"{k}={_normalize_fraction_str(v)}"
            for k, v in re.findall(r"([A-Za-z0-9_]+)=(\-?\d+(?:/[1-9]\d*)?)", body)
        )
        return {
            "kind": "causal_graph",
            "assignments": assignments,
        }

    # 3. Game-theoretic matrix tasks (Q:Game:... or domain == SYMBIOTIC_GAME)
    if compact.startswith("Q:Game:") or domain == "SYMBIOTIC_GAME":
        body = compact.split("Q:Game:", 1)[-1]
        payoffs = sorted(
            f"{k}={_normalize_fraction_str(v)}"
            for k, v in re.findall(r"([A-Za-z0-9_]+)=(\-?\d+(?:/[1-9]\d*)?)", body)
        )
        return {
            "kind": "strategic_game",
            "payoffs": payoffs,
        }

    # 4. Code execution traces (Q:Code:def f(x):return ...)
    if compact.startswith("Q:Code:") or domain == "CODE_TRACE":
        # Normalize variable name x/n/t to canonical `VAR`
        norm_code = re.sub(r"deff\(([a-zA-Z])\):return", "deff(VAR):return", compact)
        nums = [str(int(n)) for n in re.findall(r"\-?\d+", norm_code)]
        ops = re.findall(r"[\+\-\*\/\%]", norm_code)
        return {
            "kind": "program_trace",
            "operators": ops,
            "operands": nums,
        }

    # 5. General quantitative / algebraic / physics / word problem identity:
    # Strip identifier subscripts like Chain3, chain4, a0, K0, x^2 exponent marker when comparing operand tuples
    cleaned = re.sub(r"Chain\d+", "Chain", compact)
    cleaned = re.sub(r"([a-zA-Z])0=", r"\1=", cleaned)
    nums = [str(int(x)) for x in re.findall(r"\-?\d+", cleaned)]
    prefix = compact.split(":")[1] if compact.count(":") >= 2 else (domain or "GENERIC")
    return {
        "kind": "parametric_problem",
        "family": str(prefix),
        "operands": nums,
    }


def semantic_identity_hash(item: Dict[str, Any] | str) -> str:
    """Return the canonical SHA-256 hex digest of a problem's semantic identity."""
    ident = canonical_problem_identity(item)
    return hashlib.sha256(canonical_json(ident).encode("utf-8")).hexdigest()


def verify_disjoint_splits(
    splits: Dict[str, Iterable[Dict[str, Any] | str]],
) -> Dict[str, Any]:
    """Verify that all pairs of dataset splits (`train`, `validation`, `test`, etc.) have
    strictly empty intersections under `semantic_identity_hash`.

    Raises ValueError if any semantic problem identity appears in more than one split.
    """
    split_hashes: Dict[str, Set[str]] = {}
    for split_name, items in splits.items():
        if isinstance(items, (str, bytes)):
            continue
        hashes = {semantic_identity_hash(item) for item in items}
        if not hashes:
            raise ValueError(f"Split {split_name!r} is empty")
        split_hashes[split_name] = hashes

    names = sorted(split_hashes)
    overlaps: Dict[str, List[str]] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common = sorted(split_hashes[left] & split_hashes[right])
            if common:
                overlaps[f"{left}_vs_{right}"] = common

    if overlaps:
        raise ValueError(f"Semantic split leakage detected across splits: {overlaps}")

    return {
        "disjoint": True,
        "split_unique_identities": {k: len(v) for k, v in split_hashes.items()},
        "overlaps": {},
    }


def partition_by_semantic_identity(
    items: Iterable[Dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministically partition a problem pool into `train`, `validation`, and `test`
    by grouping on `semantic_identity_hash` so surface variants (e.g. `Sam` vs `Mia`,
    `p;a;b;e` vs `e;b;p;a`, `2/4` vs `1/2`) always stay in the same split.
    """
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= validation_ratio < 1.0) or (train_ratio + validation_ratio >= 1.0):
        raise ValueError("Invalid split ratios")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        h = semantic_identity_hash(item)
        groups.setdefault(h, []).append(item)

    sorted_keys = sorted(groups)
    n = len(sorted_keys)
    if n < 3:
        raise ValueError("Need at least 3 distinct semantic groups to partition into train/validation/test")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * validation_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_keys = sorted_keys[:n_train]
    val_keys = sorted_keys[n_train : n_train + n_val]
    test_keys = sorted_keys[n_train + n_val :]

    splits = {
        "train": [item for k in train_keys for item in groups[k]],
        "validation": [item for k in val_keys for item in groups[k]],
        "test": [item for k in test_keys for item in groups[k]],
    }
    verify_disjoint_splits(splits)
    return splits


SPLIT_ACCESS_POLICY: Dict[str, Dict[str, Any]] = {
    "train": {
        "allowed_consumers": ["training", "curriculum_diagnostics"],
        "includes_gold_answers": True,
        "consumes_final_test_budget": False,
    },
    "validation": {
        "allowed_consumers": ["checkpoint_selection", "hyperparameter_search", "early_stopping"],
        "includes_gold_answers": True,
        "consumes_final_test_budget": False,
    },
    "test": {
        "allowed_consumers": ["final_evaluation_scorer_only"],
        "includes_gold_answers": False,
        "consumes_final_test_budget": True,
    },
}


def build_rotating_test_partition(
    items: Iterable[Dict[str, Any]],
    *,
    epoch_id: str,
    rotation_seed: int,
    active_test_groups: int = 12,
) -> Dict[str, Any]:
    """Select a deterministic rotating final-test partition (`EVAL-04`) keyed by `epoch_id` and `rotation_seed`,
    holding out the remaining test groups in a reserve pool and stripping private `gold`/`pos`/`neg` fields
    from the public evaluation export.
    """
    if not isinstance(epoch_id, str) or not epoch_id.strip():
        raise ValueError("epoch_id must be a nonempty string")
    if active_test_groups < 1:
        raise ValueError("active_test_groups must be positive")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        h = semantic_identity_hash(item)
        groups.setdefault(h, []).append(item)

    if len(groups) < active_test_groups:
        raise ValueError("Not enough unique semantic groups for requested active_test_groups")

    # Deterministic rotation order keyed by (epoch_id, rotation_seed, group_hash)
    ranked_keys = sorted(
        groups,
        key=lambda gh: hashlib.sha256(f"{epoch_id}:{rotation_seed}:{gh}".encode("utf-8")).hexdigest(),
    )
    selected_keys = set(ranked_keys[:active_test_groups])

    public_active = []
    private_oracle = {}
    reserve_count = 0
    for gh, group_items in groups.items():
        if gh in selected_keys:
            for idx, row in enumerate(group_items):
                sample_id = row.get("sample_id") or f"{epoch_id}-{gh[:8]}-{idx:02d}"
                public_active.append(
                    {
                        "sample_id": sample_id,
                        "group_hash": gh,
                        "prompt": row["prompt"],
                        "domain": row.get("domain") or row.get("task", "GENERIC"),
                    }
                )
                private_oracle[sample_id] = str(row["gold"])
        else:
            reserve_count += len(group_items)

    return {
        "epoch_id": epoch_id,
        "rotation_seed": rotation_seed,
        "access_policy": SPLIT_ACCESS_POLICY["test"],
        "consumes_final_test": True,
        "active_group_count": len(selected_keys),
        "reserve_group_count": len(groups) - len(selected_keys),
        "reserve_prompt_count": reserve_count,
        "public_test_cases": public_active,
        "private_oracle_by_sample_id": private_oracle,
    }


def export_training_split_safe(
    splits: Dict[str, List[Dict[str, Any]]],
    *,
    requested_split: str = "train",
) -> List[Dict[str, Any]]:
    """Export training or validation data while strictly blocking final-test examples or private test answers (`EVAL-04`)."""
    if requested_split not in ("train", "validation"):
        raise PermissionError(
            f"Split {requested_split!r} cannot be exported for training or hyperparameter tuning"
        )
    verify_disjoint_splits(splits)
    return [dict(row) for row in splits[requested_split]]


def compute_difficulty_factors(item: Dict[str, Any] | str) -> Dict[str, Any]:
    """Compute measurable difficulty axes (`DATA-03` / Issue #54):
    - `operation_depth`: number of reasoning/arithmetic steps or Bayesian observations,
    - `max_denominator`: largest rational denominator (1 for integer tasks),
    - `sequence_length`: prompt character length,
    - `structure_type`: `bayesian_dag`, `causal_scm`, `normal_form_game`, `program_ast`, or `algebraic_chain`,
    - `difficulty_tier`: `tier_1_basic`, `tier_2_intermediate`, or `tier_3_compositional`.
    """
    prompt = item if isinstance(item, str) else str(item.get("prompt", ""))
    compact = re.sub(r"\s+", "", prompt)

    denominators = [int(d) for d in re.findall(r"/([1-9]\d*)", compact)]
    max_den = max(denominators) if denominators else 1

    m_bayes = re.fullmatch(r"Q:(BP|BS):([a-zA-Z0-9/=;]+);P=", compact)
    if m_bayes:
        _, body = m_bayes.groups()
        kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
        obs_len = len(kv.get("e", "1"))
        op_depth = obs_len
        structure = "bayesian_dag"
    elif compact.startswith("Q:Causal:"):
        op_depth = 2
        structure = "causal_scm"
    elif compact.startswith("Q:Game:"):
        op_depth = 2
        structure = "normal_form_game"
    elif compact.startswith("Q:Code:"):
        op_depth = max(1, len(re.findall(r"[\+\-\*\/\%]", compact)))
        structure = "program_ast"
    else:
        if "Chain4:" in compact:
            op_depth = 4
        elif any(k in compact for k in ("Chain3:", "BridgeET:", "BridgeKW:", "Prob:")):
            op_depth = 3
        else:
            op_depth = 2
        structure = "algebraic_chain"

    if op_depth >= 3 or max_den >= 5:
        tier = "tier_3_compositional"
    elif op_depth == 2 and max_den > 1:
        tier = "tier_2_intermediate"
    else:
        tier = "tier_1_basic"

    return {
        "operation_depth": op_depth,
        "max_denominator": max_den,
        "sequence_length": len(prompt),
        "structure_type": structure,
        "difficulty_tier": tier,
    }


def _independent_float_bayes_oracle(prompt: str) -> float:
    """Independent floating-point joint-table oracle for `BP` and `BS` (`DATA-04` / Issue #55).

    Instead of symbolic `Fraction` cancellation used by the generator, this oracle constructs
    the explicit 2-state joint probability table `(w_h1, w_h0)` in IEEE-754 `float64` by
    iterating over each observation character in `e`.
    """
    compact = re.sub(r"\s+", "", prompt)
    m = re.fullmatch(r"Q:(BP|BS):([a-zA-Z0-9/=;]+);P=", compact)
    if not m:
        raise ValueError(f"Unsupported Bayesian prompt for independent oracle: {prompt!r}")
    _, body = m.groups()
    kv = dict(part.split("=", 1) for part in body.split(";") if "=" in part)

    def _to_float(frac_str: str) -> float:
        if "/" in frac_str:
            n_s, d_s = frac_str.split("/", 1)
            return float(int(n_s)) / float(int(d_s))
        return float(frac_str)

    p = _to_float(kv["p"])
    a = _to_float(kv["a"])
    b = _to_float(kv["b"])
    e_seq = kv["e"]

    w1 = p
    w0 = 1.0 - p
    for ch in e_seq:
        if ch == "1":
            w1 *= a
            w0 *= b
        elif ch == "0":
            w1 *= (1.0 - a)
            w0 *= (1.0 - b)
        else:
            raise ValueError(f"Invalid observation bit {ch!r}")
    total = w1 + w0
    if total <= 0.0:
        raise ValueError("Zero marginal evidence probability in independent oracle")
    return w1 / total


def cross_check_generator_with_independent_oracle(
    items: Iterable[Dict[str, Any]],
    *,
    abs_tol: float = 1e-12,
) -> Dict[str, Any]:
    """Cross-check generator-produced items against an independent reference oracle (`DATA-04` / Issue #55)
    and detect any injected generator mistakes.
    """
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    for idx, row in enumerate(items):
        prompt = str(row["prompt"])
        gold_str = str(row["gold"])
        oracle_val = _independent_float_bayes_oracle(prompt)
        try:
            gold_val = float(Fraction(gold_str))
            in_bounds = 0.0 <= gold_val <= 1.0
            reduced = (
                f"{Fraction(gold_str).numerator}/{Fraction(gold_str).denominator}" == gold_str
                if "/" in gold_str
                else str(Fraction(gold_str).numerator) == gold_str
            )
        except Exception:
            gold_val = float("nan")
            in_bounds = False
            reduced = False

        checked += 1
        if (not in_bounds) or (not reduced) or abs(oracle_val - gold_val) > abs_tol:
            mismatches.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "generator_gold": gold_str,
                    "oracle_float": round(oracle_val, 12),
                }
            )

    return {
        "checked_count": checked,
        "mismatch_count": len(mismatches),
        "all_verified": len(mismatches) == 0,
        "mismatches": mismatches,
        "shared_untested_assumptions": [
            "Both generator and independent oracle assume binary hypothesis space H in {0, 1}.",
            "Both assume conditional independence of sequential observations given H in BS tasks.",
        ],
    }


def classify_generalization_category(item: Dict[str, Any]) -> str:
    """Classify each problem into an explicit generalization category (`DATA-06` / Issue #57):
    - `in_distribution_template`: canonical field order (`p;a;b;e`) and standard sequence length (`<=2`),
    - `representation_or_layout_novelty`: reordered field grammar (`e;b;p;a` or `novel` layout),
    - `compositional_depth_novelty`: longer sequential observation chain (`len(e) >= 3` or `longer_sequence`),
    - `parameter_extrapolation`: held-out `(p, a, b)` parameter triple in canonical layout.
    """
    layout = str(item.get("layout", ""))
    prompt = str(item.get("prompt", ""))
    diff = compute_difficulty_factors(item)
    if layout == "longer_sequence" or diff["operation_depth"] >= 3:
        return "compositional_depth_novelty"
    if layout == "novel" or prompt.startswith("Q:BP:e=") or prompt.startswith("Q:BS:e="):
        return "representation_or_layout_novelty"
    if item.get("split") in ("validation", "test"):
        return "parameter_extrapolation"
    return "in_distribution_template"


def generate_generalization_taxonomy_manifest(
    splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a generalization-category manifest across splits (`DATA-06` / Issue #57)
    and enforce that held-out grammars (`representation_or_layout_novelty`,
    `compositional_depth_novelty`) are never present in `train`.
    """
    summary: Dict[str, Dict[str, int]] = {}
    tagged_cases: Dict[str, List[Dict[str, Any]]] = {}

    for split_name, rows in splits.items():
        if not isinstance(rows, list):
            continue
        counts: Dict[str, int] = {}
        tagged_rows: List[Dict[str, Any]] = []
        for r in rows:
            row_copy = dict(r, split=split_name)
            cat = classify_generalization_category(row_copy)
            if split_name == "train" and cat in ("representation_or_layout_novelty", "compositional_depth_novelty"):
                raise ValueError(
                    f"Held-out grammar category {cat!r} is prohibited in training split"
                )
            counts[cat] = counts.get(cat, 0) + 1
            row_copy["generalization_category"] = cat
            tagged_rows.append(row_copy)
        summary[split_name] = counts
        tagged_cases[split_name] = tagged_rows

    return {
        "schema_version": "1.0.0",
        "split_category_counts": summary,
        "tagged_splits": tagged_cases,
    }


def audit_answer_frequency_shortcuts(
    train_items: List[Dict[str, Any]],
    eval_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure constant-majority, prior-only (`P(H)=p`), and prompt-length-bucketed shortcut
    baselines (`DATA-07` / Issue #64) so promotion gates cannot omit stronger trivial shortcuts.
    """
    if not train_items or not eval_items:
        raise ValueError("train_items and eval_items must be nonempty")

    from collections import Counter

    train_golds = [str(r["gold"]) for r in train_items]
    majority_ans = Counter(train_golds).most_common(1)[0][0]

    by_len: Dict[int, List[str]] = {}
    for r in train_items:
        by_len.setdefault(len(str(r["prompt"])), []).append(str(r["gold"]))
    len_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in by_len.items()}

    maj_hits = 0
    prior_hits = 0
    len_hits = 0
    for r in eval_items:
        gold = str(r["gold"])
        prompt = str(r["prompt"])
        if majority_ans == gold:
            maj_hits += 1
        m_p = re.search(r"(?:^|:|;)p=([0-9]+/[0-9]+)", prompt)
        if m_p and m_p.group(1) == gold:
            prior_hits += 1
        pred_len = len_lookup.get(len(prompt), majority_ans)
        if pred_len == gold:
            len_hits += 1

    n = len(eval_items)
    shortcuts = {
        "constant_majority": round(maj_hits / n, 6),
        "prior_only_echo": round(prior_hits / n, 6),
        "prompt_length_bucket": round(len_hits / n, 6),
    }
    strongest_name = max(shortcuts, key=shortcuts.get)
    return {
        "eval_sample_count": n,
        "majority_answer": majority_ans,
        "shortcut_accuracies": shortcuts,
        "strongest_shortcut_name": strongest_name,
        "strongest_shortcut_accuracy": shortcuts[strongest_name],
    }


def build_validation_curriculum_experiment_set(
    splits: Dict[str, List[Dict[str, Any]]],
    *,
    partition_id: str,
    max_tuning_trials: int,
) -> Dict[str, Any]:
    """Reserve a validation-only curriculum experiment set (`DATA-08` / Issue #65)
    with an explicit tuning budget and zero final-test access.
    """
    if not partition_id.strip() or max_tuning_trials < 1:
        raise ValueError("Valid partition_id and positive max_tuning_trials required")
    verify_disjoint_splits(splits)
    val_rows = export_training_split_safe(splits, requested_split="validation")
    payload = [
        {
            "prompt": r["prompt"],
            "gold": r["gold"],
            "difficulty": compute_difficulty_factors(r),
        }
        for r in val_rows
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "schema_version": "1.0.0",
        "partition_id": partition_id,
        "allowed_split": "validation",
        "consumes_final_test_budget": False,
        "max_tuning_trials": int(max_tuning_trials),
        "sample_count": len(payload),
        "partition_sha256": digest,
        "samples": payload,
    }


def build_benchmark_exclusion_ledger(
    *,
    historical_dataset_id: str,
    historical_dataset_sha256: str,
    excluded_cases: List[Dict[str, str]],
    replacement_partition_id: str,
) -> Dict[str, Any]:
    """Version benchmark exclusions and contamination findings (`DATA-13` / Issue #71)
    while preserving historical dataset SHA-256 identities.
    """
    if not excluded_cases:
        raise ValueError("excluded_cases must be nonempty")
    for c in excluded_cases:
        if not c.get("prompt") or not c.get("reason") or not c.get("semantic_identity_hash"):
            raise ValueError("Each excluded case must include prompt, semantic_identity_hash, and reason")

    return {
        "schema_version": "1.0.0",
        "historical_dataset_id": historical_dataset_id,
        "historical_dataset_sha256": historical_dataset_sha256,
        "historical_reports_retain_original_sha256": True,
        "replacement_partition_id": replacement_partition_id,
        "excluded_count": len(excluded_cases),
        "excluded_cases": [dict(c) for c in excluded_cases],
    }


def analyze_bayesian_pilot_errors(pilot_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the rejected Bayesian pilot candidates (`TRAIN-01` / Issue #76) from
    `research/bayesian-pilot/results.json` without using the consumed test set to select new hyperparameters.
    """
    task_diagnostics: Dict[str, Any] = {}
    for task_name, task_data in pilot_results.get("tasks", {}).items():
        val_loss = float(task_data["training"]["selected_validation_loss"])
        test_learned = task_data["candidate"]
        test_prior = task_data["controls"]["prior_only"]
        total = int(test_learned["total"])
        correct = int(test_learned["correct"])
        errors = total - correct
        novel_acc = float(test_learned.get("novel_layout_accuracy", 0.0))

        task_diagnostics[task_name] = {
            "promotion_status": task_data["promotion_status"],
            "selected_validation_loss": val_loss,
            "test_total": total,
            "test_correct": correct,
            "test_errors": errors,
            "prior_only_control_accuracy": float(test_prior["accuracy"]),
            "learned_minus_prior_only": round(float(test_learned["accuracy"]) - float(test_prior["accuracy"]), 6),
            "novel_layout_accuracy": novel_acc,
            "root_cause_categories": {
                "frozen_router_and_shared_backbone_bottleneck": (
                    "Only 1 routed expert (49,152 params) was trained while encoder, router, shared expert, "
                    "pointer head, and decoder remained frozen to v11 weights."
                ),
                "below_prior_only_shortcut": correct < int(test_prior["correct"]),
                "representation_layout_collapse": novel_acc <= 0.03,
            },
        }

    return {
        "schema_version": "1.0.0",
        "used_consumed_test_set_for_hyperparameter_selection": False,
        "tasks": task_diagnostics,
        "recommended_next_steps_on_validation_only": [
            "Use validation-only curriculum partition (DATA-08) for any representation or multi-layer adapter ablation.",
            "Compare candidate losses against prior-only and majority shortcuts on validation before touching held-out test slices.",
        ],
    }



