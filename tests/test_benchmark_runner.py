"""Runner behavior, adapter failures, artifact integrity and actual VeriSci CLI."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import ValidationError

from benchmark_runner import run_benchmark, save_result
from evaluation_result import validate_evaluation_result
from provider_adapters import ReplayProvider

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"provider": "recorded", "model_id": "test-model", "revision": "test-rev", "parameters": {"temperature": 0}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"sample_id": "a", "input": {"answer": "yes"}},
                      {"sample_id": "b", "input": {"answer": "no"}}]
        self.recording = {"model": MODEL, "responses": {
            "a": {"output": "yes", "input_tokens": 3, "output_tokens": 1, "cached_input_tokens": 2, "cost_usd": 0.01},
            "b": {"output": "wrong", "input_tokens": 4, "output_tokens": 1, "cost_usd": 0.02}}}

    def run_result(self, provider=None, scorer=None):
        return run_benchmark(self.cases, dataset={"name": "test", "revision": "v1", "split": "test"},
                             provider=provider or ReplayProvider(self.recording),
                             scorer=scorer or (lambda task, text: float(task["answer"] == text)),
                             scorer_metadata={"name": "exact-match", "version": "1"}, run_id="test-run")

    def test_scores_usage_costs_and_provenance(self):
        result = self.run_result()
        self.assertEqual(result["summary"], {"sample_count": 2, "mean_score": 0.5, "total_cost_usd": 0.03})
        self.assertEqual(result["model"]["revision"], "test-rev")
        self.assertEqual(result["samples"][0]["cached_input_tokens"], 2)
        self.assertEqual(result["samples"][1]["input_tokens"], 4)
        self.assertTrue(all(s["latency_ms"] >= 0 for s in result["samples"]))
        validate_evaluation_result(result)

    def test_manifest_is_stable_and_sensitive_to_input(self):
        first = self.run_result()["dataset"]["manifest_sha256"]
        self.assertEqual(first, self.run_result()["dataset"]["manifest_sha256"])
        self.cases[0]["input"]["answer"] = "changed"
        self.assertNotEqual(first, self.run_result()["dataset"]["manifest_sha256"])

    def test_duplicate_ids_fail_before_provider_is_called(self):
        provider = ReplayProvider(self.recording)
        self.cases[1]["sample_id"] = "a"
        with patch.object(provider, "generate") as generate:
            with self.assertRaisesRegex(ValueError, "unique"):
                self.run_result(provider)
            generate.assert_not_called()

    def test_provider_error_counts_as_zero_and_continues(self):
        del self.recording["responses"]["a"]
        self.recording["responses"]["b"]["output"] = "no"
        result = self.run_result()
        self.assertEqual(result["summary"]["mean_score"], 0.5)
        self.assertEqual(result["samples"][0]["error"], "provider: KeyError")
        self.assertNotIn("total_cost_usd", result["summary"])
        self.assertNotIn("error", result["samples"][1])

    def test_scorer_error_preserves_output_and_cost(self):
        def broken(task, output):
            raise ValueError("invalid submission")
        result = self.run_result(scorer=broken)
        self.assertEqual(result["summary"]["mean_score"], 0)
        self.assertEqual(result["samples"][0]["output"], "yes")
        self.assertEqual(result["samples"][0]["error"], "scorer: ValueError")
        self.assertEqual(result["summary"]["total_cost_usd"], 0.03)

    def test_invalid_adapter_usage_and_scores_fail(self):
        for field, value in (("cost_usd", float("nan")), ("cost_usd", float("inf")),
                             ("input_tokens", -1), ("output", 123)):
            with self.subTest(field=field, value=value):
                recording = copy.deepcopy(self.recording)
                recording["responses"]["a"][field] = value
                with self.assertRaises((ValueError, ValidationError)):
                    self.run_result(ReplayProvider(recording))
        with self.assertRaises(ValidationError):
            self.run_result(scorer=lambda task, text: 1.1)

    def test_unknown_usage_is_not_invented(self):
        self.recording["responses"]["a"] = {"output": "yes"}
        result = self.run_result()
        self.assertNotIn("input_tokens", result["samples"][0])
        self.assertNotIn("cost_usd", result["samples"][0])
        self.assertNotIn("total_cost_usd", result["summary"])

    def test_malformed_recording_fails_before_evaluation(self):
        for response in ({"output": "yes", "unsupported_field": 1}, {"cost_usd": 0.01}):
            with self.subTest(response=response):
                recording = copy.deepcopy(self.recording)
                recording["responses"]["a"] = response
                with self.assertRaises(TypeError):
                    ReplayProvider(recording)

    def test_invalid_model_provenance_fails_before_provider_call(self):
        provider = ReplayProvider(self.recording)
        del provider.model["model_id"]
        with patch.object(provider, "generate") as generate:
            with self.assertRaises(ValidationError):
                self.run_result(provider)
            generate.assert_not_called()

    def test_empty_run(self):
        self.cases = []
        self.assertEqual(self.run_result()["summary"]["mean_score"], 0)

    def test_invalid_result_cannot_overwrite_existing_artifact(self):
        result = self.run_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_result(result, path)
            original = path.read_bytes()
            result["summary"]["mean_score"] = 0.9
            with self.assertRaises(ValueError):
                save_result(result, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_write_failure_preserves_existing_artifact_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("original")
            with patch("benchmark_runner.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    save_result(self.run_result(), path)
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


class BenchmarkCLITests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_benchmark.py"), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_reference_and_replay_all_four_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            proc = self.cli("--output", reference)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(reference.read_text())
            validate_evaluation_result(result)
            self.assertEqual(result["summary"]["sample_count"], 8)
            self.assertEqual(result["summary"]["mean_score"], 1)
            self.assertEqual(result["summary"]["total_cost_usd"], 0)
            self.assertTrue(all("error" not in s for s in result["samples"]))
            self.assertEqual({s["sample_id"].split("-")[1] for s in result["samples"]},
                             {"logic", "silicon", "systems", "matter"})
            recording = Path(directory) / "recording.json"
            recording.write_text(json.dumps({"model": MODEL, "responses": {
                s["sample_id"]: {"output": s["output"], "cost_usd": 0.01} for s in result["samples"]}}))
            replay = Path(directory) / "replay.json"
            proc = self.cli("--provider", "replay", "--recording", recording, "--output", replay)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            replay_result = json.loads(replay.read_text())
            self.assertEqual(replay_result["summary"]["mean_score"], 1)
            self.assertEqual(replay_result["summary"]["total_cost_usd"], 0.08)
            self.assertEqual(replay_result["dataset"], result["dataset"])
            # Bad output is scored as a failure, saved, and signaled to automation.
            data = json.loads(recording.read_text())
            next(iter(data["responses"].values()))["output"] = "not JSON"
            recording.write_text(json.dumps(data))
            proc = self.cli("--provider", "replay", "--recording", recording, "--output", replay)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            failed = json.loads(replay.read_text())
            self.assertEqual(failed["summary"]["mean_score"], 7 / 8)
            self.assertIn("error", failed["samples"][0])

    def test_bad_cli_options_do_not_create_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for args in (("--samples-per-domain", "0"), ("--provider", "replay"),
                         ("--domains", "logic", "logic"), ("--difficulty", "11")):
                with self.subTest(args=args):
                    proc = self.cli("--output", output, *args)
                    self.assertEqual(proc.returncode, 2)
                    self.assertFalse(output.exists())

    def test_replay_manifest_mismatch_cannot_overwrite_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("existing")
            recording = Path(directory) / "recording.json"
            recording.write_text(json.dumps({"model": MODEL, "responses": {"extra-id": {"output": "{}"}}}))
            proc = self.cli("--provider", "replay", "--recording", recording, "--output", output)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("exactly match", proc.stderr)
            self.assertEqual(output.read_text(), "existing")

    def test_source_revision_rejects_modified_dependency(self):
        from run_benchmark import revision
        with patch("run_benchmark.subprocess.check_output", side_effect=["commit\n", " M source.py\n", ""]):
            with self.assertRaisesRegex(ValueError, "clean working tree"):
                revision("verisci")


if __name__ == "__main__":
    unittest.main()
