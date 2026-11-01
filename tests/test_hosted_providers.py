"""Authenticated hosted API contracts and costing over a local HTTP fixture."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from benchmark_runner import run_benchmark
from dataset_manifest import make_manifest, save_manifest
from evaluation_result import validate_evaluation_result
from comparison_report import compare_results
from statistical_gate import assess_comparison
from hosted_providers import HostedProvider, HostedProtocolError
from pricing import estimate_cost, validate_pricing
from run_benchmark import generate_cases, revision
from test_ollama_provider import task

ROOT = Path(__file__).resolve().parents[1]
KEY = "fixture-secret-never-log-this"


def card(provider="openai"):
    return {"pricing_version": "1.0.0", "rate_card_id": "synthetic-test-rates", "provider": provider,
            "model_id": "fixture-model", "currency": "USD", "input_per_million": 2,
            "output_per_million": 4, "cached_input_per_million": 1, "cache_write_input_per_million": 2}


def response(provider, content="{}"):
    if provider == "openai":
        return {"model": "fixture-model-snapshot", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}],
                "usage": {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 0}}}
    return {"model": "fixture-model-snapshot", "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}}


class HostedServer:
    def __enter__(self):
        self.requests, self.status, self.body, self.location = [], 200, None, None
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append((self.path, dict(self.headers), payload))
                status = fixture.status if self.headers.get("Authorization") == "Bearer " + KEY else 401
                self.send_response(status)
                if fixture.location:
                    self.send_header("Location", fixture.location)
                self.end_headers()
                provider = "openai" if self.path == "/v1/responses" else "groq"
                body = fixture.body(payload, provider) if callable(fixture.body) else fixture.body
                if body is None:
                    body = response(provider)
                self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class HostedProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HostedServer().__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__()

    def setUp(self):
        self.server.requests, self.server.status, self.server.body, self.server.location = [], 200, None, None
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": KEY, "GROQ_API_KEY": KEY})
        self.env.start()
        self.addCleanup(self.env.stop)

    def provider(self, provider="openai", **options):
        return HostedProvider(provider, "fixture-model", base_url=self.server.url, **options)

    def run_result(self, provider, count=1):
        return run_benchmark([{"sample_id": str(index), "input": task()} for index in range(count)],
                             dataset={"name": "fixture", "revision": "test", "split": "eval"},
                             provider=provider, scorer=lambda *_: 1, scorer_metadata={"name": "test", "version": "1"})

    def test_authenticated_public_requests_normalized_usage_cost_and_identity(self):
        for name, path in (("openai", "/v1/responses"), ("groq", "/openai/v1/chat/completions")):
            with self.subTest(provider=name):
                provider = self.provider(name, pricing=card(name), max_output_tokens=256)
                result = self.run_result(provider)
                validate_evaluation_result(result)
                actual_path, headers, payload = self.server.requests[-1]
                self.assertEqual(actual_path, path)
                self.assertEqual(headers["Authorization"], "Bearer " + KEY)
                self.assertNotIn("PRIVATE", json.dumps(payload))
                self.assertNotIn("reference_solution", json.dumps(payload))
                self.assertNotIn(KEY, json.dumps(result))
                self.assertNotIn("temperature", payload)  # Model defaults, unless explicitly chosen.
                self.assertFalse(payload["stream"])
                if name == "openai":
                    self.assertFalse(payload["store"])
                    self.assertEqual(payload["text"]["format"]["type"], "json_object")
                    self.assertEqual(payload["max_output_tokens"], 256)
                else:
                    self.assertEqual(payload["response_format"]["type"], "json_object")
                    self.assertEqual(payload["max_completion_tokens"], 256)
                sample = result["samples"][0]
                self.assertEqual((sample["input_tokens"], sample["output_tokens"], sample["cached_input_tokens"]), (100, 20, 40))
                self.assertAlmostEqual(sample["cost_usd"], 0.00024)
                self.assertAlmostEqual(result["summary"]["total_cost_usd"], 0.00024)
                self.assertEqual(result["model"]["revision"], "fixture-model-snapshot")
                self.assertEqual(result["model"]["parameters"]["pricing"]["rate_card_id"], "synthetic-test-rates")

    def test_absent_usage_or_pricing_does_not_invent_costs(self):
        result = self.run_result(self.provider())
        self.assertNotIn("cost_usd", result["samples"][0])
        self.server.body = response("openai")
        del self.server.body["usage"]
        result = self.run_result(self.provider(pricing=card()))
        self.assertNotIn("input_tokens", result["samples"][0])
        self.assertNotIn("total_cost_usd", result["summary"])

    def test_cache_writes_use_versioned_usage_and_their_own_price(self):
        self.server.body = response("openai")
        self.server.body["usage"]["input_tokens_details"]["cache_write_tokens"] = 20
        result = self.run_result(self.provider(pricing=dict(card(), cache_write_input_per_million=3)))
        self.assertEqual(result["schema_version"], "1.1.0")
        self.assertEqual(result["samples"][0]["cache_write_input_tokens"], 20)
        self.assertAlmostEqual(result["samples"][0]["cost_usd"], 0.00026)
        validate_evaluation_result(result)

    def test_refusal_and_truncation_preserve_usage_and_cost_at_zero_score(self):
        for provider in ("openai", "groq"):
            for reason in ("refusal", "length"):
                with self.subTest(provider=provider, reason=reason):
                    body = response(provider, "partial")
                    if provider == "openai":
                        if reason == "refusal":
                            body["output"][0]["content"] = [{"type": "refusal", "refusal": KEY}]
                        else:
                            body["status"] = "incomplete"
                    elif reason == "refusal":
                        body["choices"][0]["message"]["refusal"] = KEY
                    else:
                        body["choices"][0]["finish_reason"] = "length"
                    self.server.body = body
                    result = self.run_result(self.provider(provider, pricing=card(provider)))
                    self.assertEqual(result["summary"]["mean_score"], 0)
                    self.assertIn("error", result["samples"][0])
                    self.assertAlmostEqual(result["samples"][0]["cost_usd"], 0.00024)
                    self.assertNotIn(KEY, json.dumps(result))

    def test_http_auth_rate_limit_and_server_errors_are_redacted_and_not_retried(self):
        for status in (401, 429, 500, 503):
            self.server.requests = []
            self.server.status, self.server.body = status, {"error": {"message": KEY}}
            result = self.run_result(self.provider())
            self.assertEqual(result["samples"][0]["error"], "provider: HTTPError")
            self.assertNotIn(KEY, json.dumps(result))
            self.assertEqual(len(self.server.requests), 1)

    def test_redirect_does_not_forward_authorization(self):
        self.server.status, self.server.location = 302, self.server.url + "/stolen"
        with self.assertRaises(HTTPError):
            self.provider().generate("a", task())
        self.assertEqual(len(self.server.requests), 1)

    def test_invalid_service_data_fail_without_scoring(self):
        for body in (b"not json", [], {"error": {"message": KEY}},
                     dict(response("openai"), usage={"input_tokens": -1}),
                     dict(response("openai"), usage={"input_tokens": True}),
                     dict(response("openai"), usage={"input_tokens": 1, "input_tokens_details": {"cached_tokens": 2}})):
            self.server.body = body
            result = self.run_result(self.provider(pricing=card()))
            self.assertEqual(result["summary"]["mean_score"], 0)
            self.assertIn("error", result["samples"][0])
            self.assertNotIn(KEY, json.dumps(result))

    def test_model_drift_is_rejected_and_first_served_identity_is_retained(self):
        def changing(payload, provider):
            body = response(provider)
            body["model"] = "first" if len(self.server.requests) == 1 else "second"
            return body
        self.server.body = changing
        result = self.run_result(self.provider(), count=2)
        self.assertEqual(result["model"]["revision"], "first")
        self.assertEqual(result["samples"][1]["error"], "provider: HostedProtocolError")

    def test_setup_failures_and_timeout_forwarding(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(ValueError):
                self.provider()
        for options in ({"timeout": float("inf")}, {"max_output_tokens": 0}, {"temperature": -1}, {"pricing": card("groq")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.provider(**options)
        for origin in ("http://example.com", "https://user:secret@example.com", "https://example.com?key=secret"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                HostedProvider("openai", "fixture-model", base_url=origin)
        provider = self.provider(timeout=0.25)
        with patch.object(provider.opener, "open", side_effect=TimeoutError(KEY)) as send:
            with self.assertRaises(TimeoutError):
                provider.generate("a", task())
            self.assertEqual(send.call_args.kwargs["timeout"], 0.25)
            self.assertEqual(send.call_count, 1)

    def test_cli_both_providers_all_domains_and_missing_auth_preserves_output(self):
        from rlvr_forge import solve_reference
        from verisci import DomainTask
        generation = {"seed": 2026, "difficulty": 1, "domains": ["logic", "silicon", "systems", "matter"], "samples_per_domain": 1}
        cases = generate_cases(generation)
        answers = {case["sample_id"]: json.dumps(solve_reference(DomainTask(**case["input"]))[0]) for case in cases}
        manifest = make_manifest(cases, {"name": "rlvr-forge-procedural", "revision": revision("rlvr-forge"), "split": "eval"},
                                 {"name": "verisci-reward", "version": revision("verisci"), "configuration": generation}, generation)
        def answer(payload, provider):
            messages = payload.get("input", payload.get("messages"))
            public = json.loads(messages[1]["content"])
            self.assertNotIn("metadata", public)
            return response(provider, answers[public["task_id"]])
        self.server.body = answer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp, pp, output = root / "manifest.json", root / "prices.json", root / "result.json"
            save_manifest(manifest, mp)
            compared_results = {}
            for provider in ("openai", "groq"):
                pp.write_text(json.dumps(card(provider)))
                command = [sys.executable, str(ROOT / "run_benchmark.py"), "--provider", provider, "--model", "fixture-model",
                           "--api-url", self.server.url, "--manifest", str(mp), "--pricing", str(pp), "--output", str(output)]
                proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                result = json.loads(output.read_text())
                compared_results[provider] = result
                self.assertEqual(result["summary"]["sample_count"], 4)
                self.assertEqual(result["summary"]["mean_score"], 1)
                self.assertAlmostEqual(result["summary"]["total_cost_usd"], 0.00096)
                self.assertNotIn(KEY, output.read_text() + proc.stdout + proc.stderr)
                original = output.read_bytes()
                env = dict(os.environ, **{provider.upper() + "_API_KEY": ""})
                proc = subprocess.run(command, env=env, capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(output.read_bytes(), original)
                self.assertNotIn(KEY, proc.stderr)
            comparison = compare_results(compared_results["openai"], compared_results["groq"], manifest)
            self.assertEqual(comparison["overall"]["mean_score_delta"], 0)
            self.assertAlmostEqual(comparison["overall"]["candidate"]["cost_usd"]["total"], 0.00096)
            policy = json.loads((ROOT / "examples/regression-policy-v1.json").read_text())
            self.assertEqual(assess_comparison(comparison, policy)["decision"], "INCONCLUSIVE")


class PricingTests(unittest.TestCase):
    def test_cached_and_uncached_costs_unknown_coverage_and_flat_tariffs(self):
        rates = card()
        self.assertAlmostEqual(estimate_cost({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40}, rates), 0.00024)
        self.assertAlmostEqual(estimate_cost({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 0}, rates), 0.00028)
        self.assertIsNone(estimate_cost({"input_tokens": 100, "output_tokens": 20}, rates))
        self.assertIsNone(estimate_cost({"input_tokens": 100}, rates))
        self.assertAlmostEqual(estimate_cost({"input_tokens": 100, "output_tokens": 20}, dict(rates, cached_input_per_million=2)), 0.00028)
        self.assertIsNone(estimate_cost({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40}, dict(rates, cached_input_per_million=None)))
        self.assertIsNone(estimate_cost({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40}, dict(rates, cache_write_input_per_million=3)))
        self.assertAlmostEqual(estimate_cost({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40, "cache_write_input_tokens": 20}, dict(rates, cache_write_input_per_million=3)), 0.00026)
        self.assertIsNone(estimate_cost({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40, "cache_write_input_tokens": 20}, dict(rates, cache_write_input_per_million=None)))

    def test_invalid_price_cards_are_rejected(self):
        from jsonschema import ValidationError
        for key, value in (("input_per_million", -1), ("input_per_million", float("nan")), ("currency", "EUR"), ("pricing_version", "2")):
            with self.subTest(key=key), self.assertRaises((ValueError, ValidationError)):
                validate_pricing(dict(card(), **{key: value}))


if __name__ == "__main__":
    unittest.main()
