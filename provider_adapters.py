"""Provider boundary for local and recorded evaluation submissions.

Adapters return text plus measured usage; they never assign benchmark scores.
"""
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class ProviderResponse:
    output: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_write_input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None

    def usage(self):
        return {key: value for key, value in vars(self).items()
                if key not in ("output", "error") and value is not None}


class Provider(Protocol):
    model: dict

    def generate(self, sample_id: str, task: dict) -> ProviderResponse:
        """Return a submission; raise on a provider failure."""
        ...


class ReferenceProvider:
    """Local oracle baseline, explicitly identified as such in results."""

    def __init__(self, revision: str, private_tasks=None):
        import copy
        self.private_tasks = copy.deepcopy(private_tasks) if private_tasks is not None else None
        self.model = {"provider": "local-reference", "model_id": "rlvr-forge-reference",
                      "revision": revision, "parameters": {}, "adapter_version": "1.0.0"}

    def generate(self, sample_id, task):
        import json
        from rlvr_forge import solve_reference
        from verisci import DomainTask

        if self.private_tasks is not None:
            task = self.private_tasks[sample_id]
        submission, _ = solve_reference(DomainTask(**task))
        return ProviderResponse(json.dumps(submission, sort_keys=True), cost_usd=0.0)


class ReplayProvider:
    """Replay recorded responses keyed by sample ID, preserving model provenance."""

    def __init__(self, recording):
        import copy
        self.model = copy.deepcopy(recording["model"])
        self.model["adapter_version"] = "replay-1.0.0"
        self.responses = {
            sample_id: ProviderResponse(**copy.deepcopy(response))
            for sample_id, response in recording["responses"].items()
        }

    def generate(self, sample_id, task):
        return self.responses[sample_id]
