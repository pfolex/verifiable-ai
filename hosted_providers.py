"""Authenticated OpenAI Responses and Groq Chat adapters (EVAL-007)."""
import copy
import ipaddress
import json
import math
import os
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ollama_provider import public_task, SUBMISSION_INSTRUCTIONS
from pricing import estimate_cost, validate_pricing
from provider_adapters import ProviderResponse


class HostedProtocolError(ValueError):
    """Malformed service data; never include server error text or credentials."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward bearer credentials through redirects.


class HostedProvider:
    SETTINGS = {
        "openai": ("https://api.openai.com", "/v1/responses", "OPENAI_API_KEY"),
        "groq": ("https://api.groq.com", "/openai/v1/chat/completions", "GROQ_API_KEY"),
    }

    def __init__(self, provider, model_id, *, base_url=None, timeout=120, max_output_tokens=2048,
                 temperature=None, pricing=None):
        origin, self.path, key_env = self.SETTINGS[provider]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("a hosted model ID is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            raise ValueError("output token limit must be positive")
        if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError("temperature must be between zero and two")
        self.origin = (base_url or origin).rstrip("/")
        parsed = urlsplit(self.origin)
        try:
            loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path or not (parsed.scheme == "https" or (parsed.scheme == "http" and loopback))):
            raise ValueError("hosted API requires HTTPS, or loopback HTTP for tests, without URL credentials")
        self.api_key = os.environ.get(key_env, "")
        if not self.api_key.strip() or any(ord(char) < 33 or ord(char) > 126 for char in self.api_key):
            raise ValueError("a valid provider API key environment variable is required")
        if pricing is not None:
            validate_pricing(pricing)
            if pricing["provider"] != provider or pricing["model_id"] != model_id:
                raise ValueError("price card does not match provider and requested model")
        self.pricing = copy.deepcopy(pricing)
        self.timeout = timeout
        self.provider = provider
        self.opener = build_opener(NoRedirect())
        self.model = {"provider": provider, "model_id": model_id, "adapter_version": provider + "-1.0.0",
                      "parameters": {"max_output_tokens": max_output_tokens, "endpoint": self.origin,
                                     "output_format": "json_object"}}
        if temperature is not None:
            self.model["parameters"]["temperature"] = temperature
        if pricing is not None:
            self.model["parameters"]["pricing"] = copy.deepcopy(pricing)
            self.model["parameters"]["cost_model"] = "configured-token-rate-estimate"
        self.served_model = None

    def _usage(self, response):
        raw = response.get("usage")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise HostedProtocolError("invalid usage")
        if self.provider == "openai":
            input_key, output_key, details_key = "input_tokens", "output_tokens", "input_tokens_details"
        else:
            input_key, output_key, details_key = "prompt_tokens", "completion_tokens", "prompt_tokens_details"
        usage = {}
        for source, target in ((input_key, "input_tokens"), (output_key, "output_tokens")):
            if source in raw:
                usage[target] = raw[source]
        details = raw.get(details_key)
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise HostedProtocolError("invalid token details")
        if "cached_tokens" in details:
            usage["cached_input_tokens"] = details["cached_tokens"]
        if "cache_write_tokens" in details:
            usage["cache_write_input_tokens"] = details["cache_write_tokens"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in usage.values()):
            raise HostedProtocolError("invalid token counter")
        if "cached_input_tokens" in usage and ("input_tokens" not in usage or usage["cached_input_tokens"] > usage["input_tokens"]):
            raise HostedProtocolError("invalid cached token counter")
        if "cache_write_input_tokens" in usage and ("input_tokens" not in usage or usage["cache_write_input_tokens"] + usage.get("cached_input_tokens", 0) > usage["input_tokens"]):
            raise HostedProtocolError("invalid cache-write token counter")
        return usage

    def generate(self, sample_id, task):
        public = public_task(task)
        instructions = ("Solve the task from its public specification. Return only a JSON submission, no markdown. "
                        + SUBMISSION_INSTRUCTIONS[public["domain"]])
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(public, sort_keys=True, allow_nan=False)}]
        settings = self.model["parameters"]
        if self.provider == "openai":
            payload = {"model": self.model["model_id"], "input": messages, "store": False, "stream": False,
                       "text": {"format": {"type": "json_object"}}, "max_output_tokens": settings["max_output_tokens"]}
        else:
            payload = {"model": self.model["model_id"], "messages": messages, "stream": False,
                       "response_format": {"type": "json_object"}, "max_completion_tokens": settings["max_output_tokens"]}
        if "temperature" in settings:
            payload["temperature"] = settings["temperature"]
        request = Request(self.origin + self.path, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as handle:
            response = json.load(handle)
        if not isinstance(response, dict):
            raise HostedProtocolError("invalid response")
        model = response.get("model")
        if not isinstance(model, str) or not model:
            raise HostedProtocolError("missing served model identity")
        if self.served_model is not None and model != self.served_model:
            raise HostedProtocolError("served model changed during evaluation")
        self.served_model = model
        self.model["revision"] = model  # API-reported identity, not a weight digest.
        usage = self._usage(response)
        if self.provider == "openai":
            text, refusal, unsupported = [], False, False
            output = response.get("output", [])
            if not isinstance(output, list):
                raise HostedProtocolError("invalid output list")
            for item in output:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            if not isinstance(block.get("text"), str):
                                raise HostedProtocolError("invalid text block")
                            text.append(block["text"])
                        elif block.get("type") == "refusal":
                            refusal = True
                elif item.get("type") != "reasoning":
                    unsupported = True
            content = "".join(text)
            error = ("provider: Refusal" if refusal else "provider: UnsupportedOutput" if unsupported else
                     "provider: IncompleteResponse" if response.get("status") != "completed" or response.get("error") else None)
        else:
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise HostedProtocolError("expected one completion")
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise HostedProtocolError("invalid completion text")
            error = ("provider: Refusal" if message.get("refusal") else
                     "provider: UnsupportedOutput" if message.get("tool_calls") else
                     "provider: IncompleteResponse" if choices[0].get("finish_reason") != "stop" else None)
        cost = estimate_cost(usage, self.pricing)
        return ProviderResponse(output=content, cost_usd=cost, error=error, **usage)
