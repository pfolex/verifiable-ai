"""Live HTTP adapter contract tests using an offline Ollama test server."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from benchmark_runner import run_benchmark
from evaluation_result import validate_evaluation_result
from ollama_provider import OllamaProtocolError, OllamaProvider, public_task

ROOT = Path(__file__).resolve().parents[1]


class OllamaServer:
    """Real HTTP transport; no Ollama installation or external inference in CI."""
    def __enter__(self):
        self.requests = []
        self.response = {"model": "test:latest", "done": True,
                         "message": {"content": "{}"}, "prompt_eval_count": 15, "eval_count": 5}
        self.status = 200
        self.digest = "a" * 64
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fixture.requests.append(("GET", self.path, None))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"models": [{"name": "test:latest", "digest": fixture.digest}]}).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append(("POST", self.path, payload))
                self.send_response(fixture.status)
                self.end_headers()
                response = fixture.response(payload) if callable(fixture.response) else fixture.response
                self.wfile.write(response if isinstance(response, bytes) else json.dumps(response).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def task(domain="logic"):
    return {"task_id": "test-id", "domain": domain, "difficulty": 1, "prompt": "public problem",
            "spec": {"variables": ["P"], "premises": [["P"]], "goal_clause": ["P"],
                     "reference_solution": "PRIVATE_SPEC"},
            "metadata": {"reference_solution": "PRIVATE_METADATA", "future_key": "PRIVATE_FUTURE"},
            "future_field": "PRIVATE_TOP_LEVEL"}


class OllamaProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = OllamaServer().__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.__exit__()

    def setUp(self):
        self.fixture.requests = []
        self.fixture.digest = "a" * 64
        self.fixture.status = 200
        self.fixture.response = {"model": "test:latest", "done": True, "message": {"content": "{}"},
                                 "prompt_eval_count": 15, "eval_count": 5, "prompt_eval_cached_count": 3}

    def provider(self, **kwargs):
        return OllamaProvider("test", base_url=self.fixture.url, **kwargs)

    def test_request_excludes_privileged_fields_and_preserves_usage(self):
        provider = self.provider(temperature=0.2, num_predict=100, seed=42)
        response = provider.generate("sample", task())
        posts = [(path, payload) for method, path, payload in self.fixture.requests if method == "POST"]
        self.assertEqual(len(posts), 1)
        path, payload = posts[0]
        self.assertEqual(path, "/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"], {"temperature": 0.2, "num_predict": 100, "seed": 42})
        serialized = json.dumps(payload)
        self.assertNotIn("PRIVATE", serialized)
        self.assertNotIn("reference_solution", serialized)
        public = json.loads(payload["messages"][1]["content"])
        self.assertEqual(public["spec"]["premises"], [["P"]])
        self.assertEqual(response.output, "{}")
        self.assertEqual(response.usage(), {"input_tokens": 15, "output_tokens": 5, "cached_input_tokens": 3})
        self.assertEqual(provider.model["revision"], "a" * 64)
        self.assertEqual(provider.model["model_id"], "test:latest")

    def test_public_projection_is_independent_of_metadata(self):
        original = task()
        before = copy.deepcopy(original)
        first = public_task(original)
        original["metadata"] = {"reference_solution": "changed"}
        original["spec"]["reference_solution"] = "changed"
        self.assertEqual(first, public_task(original))
        first["spec"]["premises"].append(["Q"])
        self.assertEqual(original["spec"]["premises"], before["spec"]["premises"])

    def test_missing_usage_stays_unknown(self):
        self.fixture.response = {"model": "test:latest", "done": True, "message": {"content": "{}"}}
        response = self.provider().generate("sample", task())
        self.assertEqual(response.usage(), {})

    def test_explicit_thinking_control_is_recorded_and_sent_at_top_level(self):
        provider = self.provider(think=False)
        provider.generate("sample", task())
        payload = next(payload for method, _, payload in self.fixture.requests if method == "POST")
        self.assertIs(payload["think"], False)
        self.assertNotIn("think", payload["options"])
        self.assertIs(provider.model["parameters"]["think"], False)

    def test_invalid_service_responses(self):
        for response in (
            b"not-json", [], {"error": "PRIVATE_SERVER_ERROR"},
            {"model": "test:latest", "done": False, "message": {"content": "{}"}},
            {"model": "wrong-model", "done": True, "message": {"content": "{}"}},
            {"model": "test:latest", "done": True, "message": {"content": None}},
            {"model": "test:latest", "done": True, "message": {"content": "{}"}, "eval_count": -1},
            {"model": "test:latest", "done": True, "message": {"content": "{}"}, "eval_count": True},
        ):
            with self.subTest(response=response):
                self.fixture.response = response
                provider = self.provider()
                with self.assertRaises((OllamaProtocolError, json.JSONDecodeError)):
                    provider.generate("sample", task())

    def test_missing_model_fails_without_generation_or_download(self):
        with self.assertRaises(ValueError):
            OllamaProvider("not-installed", base_url=self.fixture.url)
        self.assertTrue(all(method == "GET" for method, _, _ in self.fixture.requests))

    def test_changed_model_digest_is_rejected(self):
        provider = self.provider()
        self.fixture.digest = "b" * 64
        with self.assertRaisesRegex(OllamaProtocolError, "digest changed"):
            provider.generate("sample", task())

    def test_timeout_is_finite_and_is_passed_to_transport(self):
        provider = self.provider(timeout=0.25)
        with patch("ollama_provider.urlopen", side_effect=TimeoutError("PRIVATE_TIMEOUT")) as send:
            with self.assertRaises(TimeoutError):
                provider.generate("sample", task())
            self.assertEqual(send.call_args.kwargs["timeout"], 0.25)
            self.assertEqual(send.call_count, 1)

    def test_invalid_configuration_is_rejected_before_http(self):
        for kwargs in ({"timeout": 0}, {"timeout": float("inf")}, {"temperature": float("nan")},
                       {"num_predict": -1}, {"num_predict": True}, {"seed": True}):
            with self.subTest(kwargs=kwargs), patch("ollama_provider.urlopen") as send:
                with self.assertRaises(ValueError):
                    self.provider(**kwargs)
                send.assert_not_called()
        for url in ("file:///tmp/model", "http://user:password@localhost", "http://localhost/api/chat", "http://localhost?key=secret"):
            with self.subTest(url=url), patch("ollama_provider.urlopen") as send:
                with self.assertRaises(ValueError):
                    OllamaProvider("test", base_url=url)
                send.assert_not_called()

    def test_http_failure_is_zero_score_and_redacted_in_result(self):
        provider = self.provider()
        self.fixture.status = 503
        self.fixture.response = {"error": "PRIVATE_ERROR_DETAILS"}
        result = run_benchmark(
            [{"sample_id": "sample", "input": task()}],
            dataset={"name": "test", "revision": "test", "split": "eval"}, provider=provider,
            scorer=lambda *_: self.fail("scorer should not run"),
            scorer_metadata={"name": "test", "version": "1"},
        )
        validate_evaluation_result(result)
        self.assertEqual(result["samples"][0]["error"], "provider: HTTPError")
        self.assertEqual(result["summary"]["mean_score"], 0)
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertEqual(sum(method == "POST" for method, _, _ in self.fixture.requests), 1)

    def test_cli_four_domains_over_http(self):
        # Ground truth is held by the fixture, not included in captured requests.
        from run_benchmark import ROOT as source_root
        from rlvr_forge import TaskForge, solve_reference
        answers = {}
        for domain in TaskForge.DOMAINS:
            generated = TaskForge(split="eval").generate(seed=2026, domain=domain, difficulty=1)
            answers[generated.task_id] = json.dumps(solve_reference(generated)[0])

        def answer(payload):
            public = json.loads(payload["messages"][1]["content"])
            return {"model": "test:latest", "done": True,
                    "message": {"content": answers[public["task_id"]]},
                    "prompt_eval_count": 100, "eval_count": 20}
        self.fixture.response = answer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "run_benchmark.py"), "--provider", "ollama", "--model", "test",
                 "--ollama-url", self.fixture.url, "--samples-per-domain", "1", "--difficulty", "1",
                 "--output", str(path)], cwd=source_root, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(path.read_text())
            validate_evaluation_result(result)
            self.assertEqual(result["summary"]["sample_count"], 4)
            self.assertEqual(result["summary"]["mean_score"], 1)
            self.assertTrue(all(sample["output_tokens"] == 20 for sample in result["samples"]))
            self.assertNotIn("total_cost_usd", result["summary"])
            posts = [payload for method, _, payload in self.fixture.requests if method == "POST"]
            self.assertEqual(len(posts), 4)
            for payload in posts:
                public = json.loads(payload["messages"][1]["content"])
                self.assertNotIn("metadata", public)
                self.assertNotIn("reference_solution", json.dumps(payload))

    def test_cli_setup_errors_preserve_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("original")
            for options in (
                ["--provider", "ollama"],
                ["--provider", "reference", "--model", "test"],
                ["--provider", "ollama", "--model", "test", "--timeout", "nan"],
                ["--provider", "ollama", "--model", "missing", "--ollama-url", self.fixture.url],
            ):
                with self.subTest(options=options):
                    proc = subprocess.run(
                        [sys.executable, str(ROOT / "run_benchmark.py"), "--output", str(path), *options],
                        cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertEqual(path.read_text(), "original")

    def test_cli_malformed_output_preserves_usage_and_signals_failure(self):
        self.fixture.response["message"]["content"] = "not JSON"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "run_benchmark.py"), "--provider", "ollama", "--model", "test",
                 "--ollama-url", self.fixture.url, "--domains", "logic", "--samples-per-domain", "1",
                 "--output", str(path)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            result = json.loads(path.read_text())
            validate_evaluation_result(result)
            self.assertEqual(result["samples"][0]["output"], "not JSON")
            self.assertEqual(result["samples"][0]["output_tokens"], 5)
            self.assertEqual(result["samples"][0]["error"], "scorer: JSONDecodeError")


if __name__ == "__main__":
    unittest.main()
