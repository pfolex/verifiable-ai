"""Versioned, explicit token-rate cost estimates; no built-in provider prices."""
from decimal import Decimal
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from artifact_io import canonical_json


def validate_pricing(card):
    canonical_json(card)
    schema = json.loads((Path(__file__).resolve().parent / "schemas/token-pricing-v1.schema.json").read_text())
    Draft202012Validator(schema).validate(card)


def estimate_cost(usage, card):
    if card is None or "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    input_rate = Decimal(str(card["input_per_million"]))
    output_rate = Decimal(str(card["output_per_million"]))
    cached_rate = card["cached_input_per_million"]
    write_rate = card["cache_write_input_per_million"]
    cached = usage.get("cached_input_tokens")
    writes = usage.get("cache_write_input_tokens")
    if cached is None:
        # A flat input tariff is insensitive to cache coverage; otherwise the
        # cost is unknown rather than assuming an unreported cache count is zero.
        if cached_rate is None or Decimal(str(cached_rate)) != input_rate:
            return None
        cached = 0
    if writes is None:
        if write_rate is None or Decimal(str(write_rate)) != input_rate:
            return None
        writes = 0
    if cached + writes > usage["input_tokens"]:
        raise ValueError("cached and cache-write tokens exceed input tokens")
    if cached and cached_rate is None:
        return None
    if writes and write_rate is None:
        return None
    cost = ((usage["input_tokens"] - cached - writes) * input_rate
            + cached * Decimal(str(cached_rate or 0))
            + writes * Decimal(str(write_rate or 0))
            + usage["output_tokens"] * output_rate) / Decimal(1_000_000)
    return float(cost)
