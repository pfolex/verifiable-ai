"""Conservative fixed-horizon gates, including insufficient-data and hidden regressions."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from comparison_report import _group
from statistical_gate import assess_comparison, paired_interval
from test_comparison_report import results
from comparison_report import compare_results
from dataset_manifest import make_manifest
from run_benchmark import generate_cases
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def synthetic_report(deltas_by_domain):
    manifest, b, c = results()
    report = compare_results(b, c, manifest)
    pairs, left, right = [], {}, {}
    for domain, deltas in deltas_by_domain.items():
        for index, delta in enumerate(deltas):
            sample_id = f"{domain}-{index}"
            bs, cs = max(0, -delta), max(0, delta)
            left[sample_id] = {"sample_id": sample_id, "output": "", "score": bs, "latency_ms": 1}
            right[sample_id] = {"sample_id": sample_id, "output": "", "score": cs, "latency_ms": 1}
            pairs.append({"sample_id": sample_id, "domain": domain, "baseline_score": bs, "candidate_score": cs,
                          "delta": delta, "baseline_error": False, "candidate_error": False})
    report["pairs"] = pairs
    report["overall"] = _group(pairs, left, right)
    report["domains"] = {domain: _group([p for p in pairs if p["domain"] == domain], left, right) for domain in deltas_by_domain}
    return report


class StatisticalGateTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "examples/regression-policy-v1.json").read_text())
        self.policy["independent_samples"] = True

    def test_known_interval_formula_and_simultaneous_correction(self):
        one = paired_interval([0.2] * 100, alpha=0.05, comparisons=1)
        self.assertAlmostEqual(one["lower"], 0.2 - math.sqrt(2 * math.log(40) / 100))
        many = paired_interval([0.2] * 100, alpha=0.05, comparisons=5)
        self.assertLess(many["lower"], one["lower"])
        self.assertGreater(many["upper"], one["upper"])

    def test_clear_improvement_passes_and_regression_fails(self):
        improved = synthetic_report({"logic": [1.0] * 100, "matter": [1.0] * 100})
        self.assertEqual(assess_comparison(improved, self.policy)["decision"], "PASS")
        regressed = synthetic_report({"logic": [-1.0] * 100})
        self.assertEqual(assess_comparison(regressed, self.policy)["decision"], "FAIL")

    def test_domain_regression_cannot_hide_in_overall_improvement(self):
        report = synthetic_report({"logic": [1.0] * 1000, "matter": [-1.0] * 100})
        self.assertGreater(report["overall"]["mean_score_delta"], 0)
        gate = assess_comparison(report, self.policy)
        self.assertEqual(gate["criteria"]["matter"]["decision"], "FAIL")
        self.assertEqual(gate["decision"], "FAIL")

    def test_smoke_samples_and_undeclared_independence_cannot_pass(self):
        for report in (synthetic_report({"logic": [1.0]}), synthetic_report({"logic": [0.0] * 100})):
            self.assertEqual(assess_comparison(report, self.policy)["decision"], "INCONCLUSIVE")
        self.policy["independent_samples"] = False
        self.assertEqual(assess_comparison(synthetic_report({"logic": [1.0] * 100}), self.policy)["decision"], "INCONCLUSIVE")

    def test_policy_validation_rejects_invalid_and_nonfinite_values(self):
        report = synthetic_report({"logic": [1.0] * 100})
        for key, value in (("alpha", 0), ("alpha", float("nan")), ("min_samples_per_domain", 1),
                           ("non_regression_margin", -1), ("max_error_rate", 2)):
            policy = dict(self.policy, **{key: value})
            with self.subTest(key=key), self.assertRaises((ValueError, ValidationError)):
                assess_comparison(report, policy)

    def test_error_policy_failure_is_independent_of_sample_size(self):
        manifest, baseline, candidate = results()
        candidate["samples"][0]["error"] = "provider: TimeoutError"
        candidate["samples"][0]["score"] = 0
        candidate["summary"]["mean_score"] = 0.5625
        report = compare_results(baseline, candidate, manifest)
        self.assertEqual(assess_comparison(report, self.policy)["decision"], "FAIL")

    def test_cli_exit_codes_and_invalid_inputs_preserve_old_report(self):
        manifest, baseline, candidate = results()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bp, cp, mp, pp, output = [root / name for name in ("b.json", "c.json", "m.json", "p.json", "gate.json")]
            for path, value in ((bp, baseline), (cp, candidate), (mp, manifest), (pp, self.policy)):
                path.write_text(json.dumps(value))
            command = [sys.executable, str(ROOT / "gate_benchmarks.py"), str(bp), str(cp), "--manifest", str(mp), "--policy", str(pp), "--output", str(output)]
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(json.loads(output.read_text())["decision"], "INCONCLUSIVE")
            candidate["samples"][0]["error"] = "provider: TimeoutError"
            candidate["samples"][0]["score"] = 0
            candidate["summary"]["mean_score"] = 0.5625
            cp.write_text(json.dumps(candidate))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            original = output.read_bytes()
            self.policy["alpha"] = 0
            pp.write_text(json.dumps(self.policy))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 3)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_pass_exit_code_on_synthetic_high_confidence_policy(self):
        # A deliberately weak alpha exercises CLI plumbing, not real quality.
        manifest, baseline, candidate = results()
        generation = dict(manifest["generation"], domains=["logic"], samples_per_domain=3)
        manifest = make_manifest(generate_cases(generation), manifest["dataset"],
                                 dict(manifest["scorer"], configuration=generation), generation)
        for result, score in ((baseline, 0), (candidate, 1)):
            sample = result["samples"][0]
            result["samples"] = [dict(sample, sample_id=case["sample_id"], score=score) for case in manifest["cases"]]
            result["dataset"] = dict(manifest["dataset"], manifest_sha256=manifest["manifest_sha256"])
            result["scorer"] = manifest["scorer"]
            result["summary"] = {"sample_count": 3, "mean_score": score, "total_cost_usd": 0.03}
        policy = dict(self.policy, alpha=0.99, min_samples_per_domain=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("b", baseline), ("c", candidate), ("m", manifest), ("p", policy)):
                (root / name).write_text(json.dumps(value))
            proc = subprocess.run([sys.executable, str(ROOT / "gate_benchmarks.py"), str(root / "b"), str(root / "c"),
                                   "--manifest", str(root / "m"), "--policy", str(root / "p"), "--output", str(root / "out")],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads((root / "out").read_text())["decision"], "PASS")


if __name__ == "__main__":
    unittest.main()
