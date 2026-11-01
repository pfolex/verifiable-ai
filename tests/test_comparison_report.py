"""Comparisons must pair identical tasks and retain error/unknown-resource data."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark_runner import run_benchmark
from comparison_report import compare_results, markdown_report, validate_comparison
from dataset_manifest import load_manifest
from provider_adapters import ReplayProvider

ROOT = Path(__file__).resolve().parents[1]


def results():
    manifest = load_manifest(ROOT / "examples/dataset-manifest-v1.json")
    responses = {case["sample_id"]: {"output": "{}", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01}
                 for case in manifest["cases"]}
    provider = ReplayProvider({"model": {"provider": "test", "model_id": "test", "parameters": {}}, "responses": responses})
    def run(score):
        return run_benchmark(manifest["cases"], dataset=manifest["dataset"], provider=provider,
                             scorer=lambda *_: score, scorer_metadata=manifest["scorer"], manifest=manifest)
    return manifest, run(0.5), run(0.75)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.baseline, self.candidate = results()

    def compare(self):
        return compare_results(self.baseline, self.candidate, self.manifest)

    def test_paired_overall_domain_resources_and_latency(self):
        report = self.compare()
        self.assertEqual(report["overall"]["sample_count"], 4)
        self.assertEqual(report["overall"]["mean_score_delta"], 0.25)
        self.assertEqual(report["overall"]["wins"], 4)
        self.assertEqual(len(report["domains"]), 4)
        self.assertTrue(all(group["mean_score_delta"] == 0.25 for group in report["domains"].values()))
        self.assertEqual(report["overall"]["candidate"]["cost_usd"]["total"], 0.04)
        self.assertEqual(report["overall"]["candidate"]["tokens"]["input_tokens"]["total"], 40)
        self.assertGreaterEqual(report["overall"]["candidate"]["latency_ms"]["p95"], 0)
        self.candidate["samples"].reverse()
        self.assertEqual(self.compare()["pairs"], report["pairs"])
        self.assertIn("+0.250000", markdown_report(report))

    def test_errors_and_unknown_resources_are_not_dropped(self):
        sample = self.candidate["samples"][0]
        sample.update(score=0, error="provider: TimeoutError")
        del sample["cost_usd"]
        del self.candidate["summary"]["total_cost_usd"]
        del sample["input_tokens"]
        self.candidate["summary"].update(mean_score=0.5625)
        report = self.compare()
        metrics = report["overall"]["candidate"]
        self.assertEqual(metrics["error_count"], 1)
        self.assertEqual(report["overall"]["losses"], 1)
        self.assertIsNone(metrics["cost_usd"]["total"])
        self.assertEqual(metrics["cost_usd"]["known_samples"], 3)
        self.assertAlmostEqual(metrics["cost_usd"]["known_total"], 0.03)
        self.assertIn("unknown", markdown_report(report))

    def test_provenance_and_missing_samples_fail_closed(self):
        original = copy.deepcopy(self.candidate)
        for mutation in (
            lambda r: r["dataset"].update(manifest_sha256="0" * 64),
            lambda r: r["dataset"].update(revision="wrong"),
            lambda r: r["scorer"].update(version="wrong"),
            lambda r: r["samples"][0].update(sample_id="wrong"),
        ):
            self.candidate = copy.deepcopy(original)
            mutation(self.candidate)
            with self.assertRaises(ValueError):
                self.compare()

    def test_incorrect_report_statistics_are_rejected(self):
        for mutate in (lambda r: r["pairs"][0].update(delta=0.5),
                       lambda r: r["overall"].update(sample_count=5),
                       lambda r: r["overall"]["candidate"].update(mean_score=0.9),
                       lambda r: r["overall"].update(wins=0)):
            report = self.compare()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_comparison(report)

    def test_cli_emits_reports_and_preserves_inputs_on_bad_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate, report, md = [root / name for name in ("b.json", "c.json", "report.json", "report.md")]
            baseline.write_text(json.dumps(self.baseline))
            candidate.write_text(json.dumps(self.candidate))
            command = [sys.executable, str(ROOT / "compare_benchmarks.py"), str(baseline), str(candidate),
                       "--manifest", str(ROOT / "examples/dataset-manifest-v1.json"), "--output", str(report), "--markdown", str(md)]
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(report.read_text())["overall"]["mean_score_delta"], 0.25)
            self.assertIn("Evaluation comparison", md.read_text())
            original = report.read_bytes()
            self.candidate["scorer"]["version"] = "bad"
            candidate.write_text(json.dumps(self.candidate))
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(report.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
