"""Live Ollama adapter with a public-task-only prompt boundary."""
import copy
import json
import math
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from provider_adapters import ProviderResponse

# Only verifier inputs are public. Generator metadata and reference solutions
# remain on the scorer side, even if the generator later adds metadata fields.
PUBLIC_SPEC_KEYS = {
    "logic": ("variables", "premises", "goal_clause", "require_sat"),
    "silicon": ("inputs", "outputs", "truth_table", "max_gates", "max_depth"),
    "systems": ("num_workers", "required_commits", "resources", "max_buffer"),
    "matter": ("formula", "required_motifs", "expected_rings"),
}
SUBMISSION_INSTRUCTIONS = {
    "logic": 'Return {"proof_steps": [{"p1": integer, "p2": integer, "conclusion": [literal]}], '
             '"assignment": {variable: boolean}}. Premises are indexed from zero; each new conclusion '
             'is appended to that list. Use sound propositional resolution. Negation uses ~.',
    "silicon": 'Return {"gates": [{"op": gate, "ins": [wire], "out": wire}]}. '
               'Gates are topologically ordered and may use AND, OR, XOR, NOT, NAND, NOR, BUF or MUX. '
               'MUX input order is [select, input0, input1]. '
               'Output wire names must match the spec; obey gate and depth limits.',
    "systems": 'Return {"events": [{"op": operation, "worker": integer, "clock": integer, '
               '"resource": resource, "msg_id": message_id}]}. Operations are acquire, '
               'critical_commit, release, send, recv. Include resource for locks and msg_id '
               'for messages. Hold every required lock for a commit, advance Lamport clocks, '
               'give each worker a commit, release locks, and drain messages.',
    "matter": 'Return {"atoms": [element], "bonds": [{"u": atom_index, "v": atom_index, '
              '"order": integer}]}. Atom indices start at zero. Match formula, motifs and rings; '
              'use connected graphs with standard valences C=4, H=1, O=2, N=3.',
}


def public_task(task):
    """Project a trusted generated task onto the documented public input fields."""
    domain = task["domain"]
    return {
        "task_id": task["task_id"], "domain": domain,
        "difficulty": task["difficulty"], "prompt": task["prompt"],
        "spec": {key: copy.deepcopy(task["spec"][key])
                 for key in PUBLIC_SPEC_KEYS[domain] if key in task["spec"]},
    }


class OllamaProtocolError(ValueError):
    """The service did not return a complete, valid chat response."""


class OllamaProvider:
    def __init__(self, model_id, *, base_url="http://localhost:11434", timeout=120,
                 temperature=0, num_predict=2048, seed=0, think=None):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("a model ID is required")
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Ollama URL must be an HTTP(S) origin without credentials")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
            raise ValueError("num_predict must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if think is not None and not isinstance(think, bool):
            raise ValueError("think must be a boolean or None")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.think = think
        self.model = {
            "provider": "ollama", "model_id": model_id,
            "parameters": {"temperature": temperature, "num_predict": num_predict, "seed": seed},
            "adapter_version": "ollama-chat-1.0.0",
        }
        if think is not None:
            self.model["parameters"]["think"] = think
        # Resolve a real installed model before starting the run. Never pull a
        # missing model implicitly or invent a source revision for a mutable tag.
        match = self._installed_model(model_id)
        self.model["model_id"] = match["name"]
        self.model["revision"] = match["digest"]

    def _installed_model(self, model_id):
        models = self._request("/api/tags")["models"]
        canonical_name = model_id if ":" in model_id.rsplit("/", 1)[-1] else model_id + ":latest"
        match = next((model for model in models if model["name"] == canonical_name), None)
        if match is None or not isinstance(match.get("digest"), str) or not match["digest"]:
            raise ValueError("requested model is not installed or has no digest")
        return match

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or "error" in result:
            raise OllamaProtocolError("invalid Ollama response")
        return result

    def generate(self, sample_id, task):
        public = public_task(task)
        payload = {
            "model": self.model["model_id"], "stream": False, "format": "json",
            "options": {key: self.model["parameters"][key] for key in ("temperature", "num_predict", "seed")},
            "messages": [
                {"role": "system", "content": "Solve the task using only the supplied public specification. "
                 "Return only a JSON submission object, without markdown or commentary. "
                 + SUBMISSION_INSTRUCTIONS[public["domain"]]},
                {"role": "user", "content": json.dumps(public, sort_keys=True, allow_nan=False)},
            ],
        }
        if self.think is not None:
            payload["think"] = self.think
        result = self._request("/api/chat", payload)
        if self._installed_model(self.model["model_id"])["digest"] != self.model["revision"]:
            raise OllamaProtocolError("model digest changed during evaluation")
        if result.get("done") is not True or result.get("model") != self.model["model_id"]:
            raise OllamaProtocolError("incomplete response or unexpected model")
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaProtocolError("missing text output")
        usage = {}
        for source, destination in (("prompt_eval_count", "input_tokens"), ("eval_count", "output_tokens"),
                                    ("prompt_eval_cached_count", "cached_input_tokens")):
            if source in result:
                value = result[source]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise OllamaProtocolError("invalid usage counters")
                usage[destination] = value
        # A locally hosted model has no API price attached. Do not invent cost
        # or treat unavailable token counters as zero.
        return ProviderResponse(output=message["content"], **usage)
