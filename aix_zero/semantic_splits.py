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

