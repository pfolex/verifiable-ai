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


def validate_metamorphic_relations(
    item: Dict[str, Any],
    *,
    assume_conditional_iid: bool = True,
) -> Dict[str, Any]:
    """Validate metamorphic relations (`DATA-05` / Issue #63):
    1. Hypothesis complement symmetry: swapping `p -> 1-p`, `a -> b`, `b -> a` transforms posterior `P -> 1-P`.
    2. Observation sequence permutation invariance (for `BS`) under conditional IID: permuting `e` preserves `P`.
    3. Rejects sequence permutation if `assume_conditional_iid=False`.
    """
    if not assume_conditional_iid:
        raise ValueError("Cannot apply IID sequence-permutation metamorphic relation when assume_conditional_iid=False")

    prompt = str(item["prompt"])
    gold_frac = Fraction(str(item["gold"]))
    if not (0 < gold_frac < 1):
        raise ValueError("Metamorphic Bayesian check requires interior probability in (0, 1)")

    compact = re.sub(r"\s+", "", prompt)
    m = re.fullmatch(r"Q:(BP|BS):([a-zA-Z0-9/=;]+);P=", compact)
    if not m:
        raise ValueError(f"Unsupported prompt for metamorphic check: {prompt!r}")
    task_kind, body = m.groups()
    kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)

    p_f = Fraction(kv["p"])
    a_f = Fraction(kv["a"])
    b_f = Fraction(kv["b"])
    e_str = kv["e"]

    # 1. Hypothesis complement transform: H' = not H -> p' = 1 - p, a' = b, b' = a -> P(H'|E) = 1 - P(H|E)
    comp_p = 1 - p_f
    comp_prompt = f"Q:{task_kind}:p={comp_p.numerator}/{comp_p.denominator};a={b_f.numerator}/{b_f.denominator};b={a_f.numerator}/{a_f.denominator};e={e_str};P="
    comp_oracle = _independent_float_bayes_oracle(comp_prompt)
    expected_comp = float(1 - gold_frac)
    complement_ok = abs(comp_oracle - expected_comp) <= 1e-12

    # 2. Sequence permutation invariance under conditional IID
    perm_e = e_str[::-1]
    perm_prompt = f"Q:{task_kind}:p={p_f.numerator}/{p_f.denominator};a={a_f.numerator}/{a_f.denominator};b={b_f.numerator}/{b_f.denominator};e={perm_e};P="
    perm_oracle = _independent_float_bayes_oracle(perm_prompt)
    permutation_ok = abs(perm_oracle - float(gold_frac)) <= 1e-12

    return {
        "complement_symmetry_passed": complement_ok,
        "sequence_permutation_passed": permutation_ok,
        "all_metamorphic_relations_passed": complement_ok and permutation_ok,
        "complement_prompt": comp_prompt,
        "permuted_prompt": perm_prompt,
    }


def encode_rational_representation(frac_str: str, mode: str = "reduced_fraction") -> str:
    """Encode an exact probability/rational in one of three controlled representations (`DATA-09` / Issue #66)."""
    f = Fraction(frac_str.strip())
    if mode == "reduced_fraction":
        return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
    if mode == "num_den_fields":
        return f"num={f.numerator};den={f.denominator}"
    if mode == "scaled_integer_bps":
        scaled = f * 10000
        if scaled.denominator != 1:
            raise ValueError(f"Fraction {frac_str!r} cannot be represented exactly in basis points (1/10000)")
        return f"bps={scaled.numerator}"
    raise ValueError(f"Unsupported rational representation mode: {mode!r}")


def decode_rational_representation(encoded: str, mode: str = "reduced_fraction") -> str:
    """Decode a rational representation back to canonical reduced fraction string (`DATA-09` / Issue #66)."""
    s = encoded.strip()
    if mode == "reduced_fraction":
        f = Fraction(s)
    elif mode == "num_den_fields":
        m = re.fullmatch(r"num=(\-?\d+);den=([1-9]\d*)", s)
        if not m:
            raise ValueError(f"Malformed num_den_fields encoding: {s!r}")
        f = Fraction(int(m.group(1)), int(m.group(2)))
    elif mode == "scaled_integer_bps":
        m = re.fullmatch(r"bps=(\-?\d+)", s)
        if not m:
            raise ValueError(f"Malformed scaled_integer_bps encoding: {s!r}")
        f = Fraction(int(m.group(1)), 10000)
    else:
        raise ValueError(f"Unsupported rational representation mode: {mode!r}")
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def verify_multistep_bayesian_derivation(
    prompt: str,
    step_posteriors: List[str],
    final_answer: str,
) -> Dict[str, Any]:
    """Verify each sequential Bayesian update step independently (`DATA-10` / Issue #67).

    Catches an incorrect intermediate step even when `final_answer` matches the final posterior.
    """
    compact = re.sub(r"\s+", "", prompt)
    m = re.fullmatch(r"Q:(BP|BS):([a-zA-Z0-9/=;]+);P=", compact)
    if not m:
        raise ValueError(f"Unsupported Bayesian prompt: {prompt!r}")
    task_kind, body = m.groups()
    kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
    e_str = kv["e"]
    if len(step_posteriors) != len(e_str):
        raise ValueError(f"Expected {len(e_str)} step posteriors, got {len(step_posteriors)}")

    step_results: List[Dict[str, Any]] = []
    for i in range(1, len(e_str) + 1):
        prefix_e = e_str[:i]
        sub_prompt = f"Q:{task_kind}:p={kv['p']};a={kv['a']};b={kv['b']};e={prefix_e};P="
        oracle_val = _independent_float_bayes_oracle(sub_prompt)
        cand_f = Fraction(step_posteriors[i - 1].strip())
        ok = abs(float(cand_f) - oracle_val) <= 1e-12
        step_results.append(
            {
                "step_index": i,
                "observation_prefix": prefix_e,
                "candidate_posterior": f"{cand_f.numerator}/{cand_f.denominator}",
                "step_verified": ok,
            }
        )

    final_oracle = _independent_float_bayes_oracle(prompt)
    final_f = Fraction(final_answer.strip())
    final_ok = abs(float(final_f) - final_oracle) <= 1e-12
    all_steps_ok = all(s["step_verified"] for s in step_results)

    if final_ok and all_steps_ok:
        status = "VALID_DERIVATION_AND_FINAL_ANSWER"
    elif final_ok and not all_steps_ok:
        status = "INTERMEDIATE_STEP_MISMATCH_DESPITE_CORRECT_FINAL"
    else:
        status = "FINAL_ANSWER_MISMATCH"

    return {
        "supervision_mode": "derivation_supervised",
        "status": status,
        "all_steps_verified": all_steps_ok,
        "final_answer_verified": final_ok,
        "steps": step_results,
    }


def generate_interpretable_negative_examples(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate verified negative examples with interpretable error categories (`DATA-11` / Issue #68).

    Ensures no negative example is mathematically equal to the true `gold` answer.
    """
    prompt = str(item["prompt"])
    gold_f = Fraction(str(item["gold"]))
    compact = re.sub(r"\s+", "", prompt)
    m = re.fullmatch(r"Q:(BP|BS):([a-zA-Z0-9/=;]+);P=", compact)
    if not m:
        raise ValueError(f"Unsupported prompt: {prompt!r}")
    _, body = m.groups()
    kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
    p_f = Fraction(kv["p"])
    a_f = Fraction(kv["a"])
    b_f = Fraction(kv["b"])
    e_bit = kv["e"][0]

    like_h1 = a_f if e_bit == "1" else (1 - a_f)
    like_h0 = b_f if e_bit == "1" else (1 - b_f)

    candidates = [
        ("PRIOR_ONLY_BASE_RATE_COPY", p_f),
        ("UNNORMALIZED_JOINT_NUMERATOR", p_f * like_h1),
        ("EVIDENCE_COMPLEMENT_CONFUSION", (p_f * (1 - like_h1)) / (p_f * (1 - like_h1) + (1 - p_f) * (1 - like_h0))),
        ("ARITHMETIC_DENOMINATOR_OFF_BY_ONE", Fraction(gold_f.numerator, gold_f.denominator + 1)),
    ]

    negatives: List[Dict[str, str]] = []
    for err_type, val_f in candidates:
        if val_f == gold_f:
            val_f = Fraction(gold_f.numerator + 1, gold_f.denominator + 2)
        negatives.append(
            {
                "prompt": prompt,
                "correct_gold": f"{gold_f.numerator}/{gold_f.denominator}",
                "negative_answer": f"{val_f.numerator}/{val_f.denominator}",
                "error_category": err_type,
            }
        )
    return negatives


def validate_contributor_dataset_submission(package: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an offline contributor dataset submission (`DATA-12` / Issue #70)
    before it can enter any training or evaluation queue.
    """
    allowed_licenses = {"Apache-2.0", "MIT", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0"}
    lic = str(package.get("license_spdx", "")).strip()
    if lic not in allowed_licenses:
        raise ValueError(f"Rejected submission: unsupported or missing license_spdx {lic!r}")
    if not str(package.get("generator_module_or_script", "")).strip():
        raise ValueError("Rejected submission: missing generator_module_or_script provenance")

    splits = package.get("splits")
    if not isinstance(splits, dict) or not {"train", "validation", "test"}.issubset(splits):
        raise ValueError("Rejected submission: package must provide disjoint train, validation, and test splits")

    verify_disjoint_splits(splits)
    for split_name in ("train", "validation", "test"):
        rows = splits[split_name]
        hashes = [semantic_identity_hash(r) for r in rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"Rejected submission: duplicate semantic problems inside {split_name!r} split")
        oracle_check = cross_check_generator_with_independent_oracle(rows)
        if not oracle_check["all_verified"]:
            raise ValueError(f"Rejected submission: independent oracle failed on {split_name!r} split")

    return {
        "accepted_for_queue": True,
        "dataset_id": package.get("dataset_id", "contributor-submission"),
        "license_spdx": lic,
        "split_counts": {k: len(splits[k]) for k in ("train", "validation", "test")},
    }


def build_reproducible_data_inspection_report(
    splits: Dict[str, List[Dict[str, Any]]],
    *,
    dataset_id: str,
) -> Dict[str, Any]:
    """Generate a reproducible data inspection report (`DATA-14` / Issue #72) containing
    split sizes, prompt length distributions, difficulty tiers, and shortcut accuracies,
    while omitting all withheld final-test gold answers from public summaries.
    """
    verify_disjoint_splits(splits)
    shortcuts = audit_answer_frequency_shortcuts(splits["train"], splits["test"])

    split_summaries: Dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        rows = splits[split_name]
        lengths = [len(str(r["prompt"])) for r in rows]
        tiers: Dict[str, int] = {}
        for r in rows:
            t = compute_difficulty_factors(r)["difficulty_tier"]
            tiers[t] = tiers.get(t, 0) + 1
        summary_entry: Dict[str, Any] = {
            "sample_count": len(rows),
            "min_prompt_length": min(lengths),
            "max_prompt_length": max(lengths),
            "difficulty_tier_counts": tiers,
        }
        if split_name != "test":
            from collections import Counter
            summary_entry["top_answers"] = dict(Counter(str(r["gold"]) for r in rows).most_common(3))
        split_summaries[split_name] = summary_entry

    return {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "contains_withheld_test_answers": False,
        "splits": split_summaries,
        "shortcut_audit": shortcuts,
    }


def build_cross_domain_composition_pilot() -> Dict[str, Any]:
    """Combine two verified primitives (`BP` Bayesian posterior + 2-action expected utility)
    into a held-out cross-domain composition family (`DATA-15` / Issue #74) with an independent
    reference verifier.
    """
    cases = [
        {"p": "1/2", "a": "2/3", "b": "1/3", "e": "1", "u_h1": 9, "u_h0": 3},
        {"p": "1/3", "a": "3/4", "b": "1/4", "e": "1", "u_h1": 10, "u_h0": 2},
        {"p": "2/5", "a": "1/2", "b": "1/4", "e": "0", "u_h1": 8, "u_h0": 4},
    ]
    built: List[Dict[str, Any]] = []
    for c in cases:
        bp_prompt = f"Q:BP:p={c['p']};a={c['a']};b={c['b']};e={c['e']};P="
        post_float = _independent_float_bayes_oracle(bp_prompt)
        # Exact Fraction computation for gold
        p_f, a_f, b_f = Fraction(c["p"]), Fraction(c["a"]), Fraction(c["b"])
        l1 = a_f if c["e"] == "1" else (1 - a_f)
        l0 = b_f if c["e"] == "1" else (1 - b_f)
        post_f = (p_f * l1) / (p_f * l1 + (1 - p_f) * l0)
        eu_f = post_f * int(c["u_h1"]) + (1 - post_f) * int(c["u_h0"])
        eu_oracle = post_float * float(c["u_h1"]) + (1.0 - post_float) * float(c["u_h0"])
        if abs(float(eu_f) - eu_oracle) > 1e-12:
            raise AssertionError("Independent composition oracle mismatch")

        comp_prompt = (
            f"Q:COMPOSE_BP_EU:p={c['p']};a={c['a']};b={c['b']};e={c['e']};"
            f"u1={c['u_h1']};u0={c['u_h0']};EU="
        )
        gold_str = f"{eu_f.numerator}/{eu_f.denominator}" if eu_f.denominator != 1 else str(eu_f.numerator)
        built.append(
            {
                "prompt": comp_prompt,
                "primitive_1_posterior": f"{post_f.numerator}/{post_f.denominator}",
                "gold": gold_str,
                "primitives": ["BP_BAYESIAN_POSTERIOR", "EXPECTED_UTILITY_LINEAR"],
                "generalization_type": "cross_domain_two_primitive_composition",
            }
        )

    return {
        "family_id": "compose-bp-expected-utility-v1",
        "independent_oracle_verified": True,
        "cases": built,
    }




