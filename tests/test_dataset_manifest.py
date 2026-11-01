"""Manifest integrity, public data export and verified CLI reruns."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from jsonschema import ValidationError
from benchmark_runner import run_benchmark
from dataset_manifest import load_manifest, manifest_digest, make_manifest, save_manifest, validate_manifest
from run_benchmark import generate_cases, revision

ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generation = {"seed": 2026, "difficulty": 1, "domains": ["logic", "silicon", "systems", "matter"], "samples_per_domain": 1}
        cls.cases = generate_cases(cls.generation)
        cls.manifest = make_manifest(cls.cases,
            {"name": "rlvr-forge-procedural", "revision": revision("rlvr-forge"), "split": "eval"},
            {"name": "verisci-reward", "version": revision("verisci"), "configuration": cls.generation}, cls.generation)

    def test_public_round_trip_and_stable_hash(self):
        self.assertNotIn("reference_solution", json.dumps(self.manifest))
        self.assertNotIn("metadata", json.dumps(self.manifest))
        shuffled_keys = json.loads(json.dumps(self.manifest, sort_keys=True))
        self.assertEqual(manifest_digest(shuffled_keys), self.manifest["manifest_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_manifest(self.manifest, path)
            self.assertEqual(load_manifest(path), self.manifest)

    def test_tampering_duplicate_ids_unknown_fields_and_wrong_types_are_rejected(self):
        variants = []
        changed = copy.deepcopy(self.manifest)
        changed["cases"][0]["input"]["prompt"] = "changed"
        variants.append(changed)
        for change in (lambda m: m["cases"].append(m["cases"][0]),
                       lambda m: m["cases"][0]["input"].update(metadata={"reference_solution": "private"}),
                       lambda m: m["cases"][0]["input"]["spec"].update(reference_solution="private"),
                       lambda m: m["cases"][0]["input"]["spec"].update(premises=123),
                       lambda m: m["cases"][0].update(sample_id="wrong-id")):
            changed = copy.deepcopy(self.manifest)
            change(changed)
            changed["manifest_sha256"] = manifest_digest(changed)
            variants.append(changed)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises((ValueError, ValidationError)):
                validate_manifest(variant)

    def test_bad_manifest_cannot_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("original")
            bad = copy.deepcopy(self.manifest)
            bad["dataset"]["revision"] = "tampered"
            with self.assertRaises(ValueError):
                save_manifest(bad, path)
            self.assertEqual(path.read_text(), "original")

    def test_runner_rejects_manifest_input_mismatch_before_provider_call(self):
        cases = copy.deepcopy(self.manifest["cases"])
        cases[0]["input"]["prompt"] = "changed"
        provider = Mock()
        with self.assertRaisesRegex(ValueError, "do not match manifest"):
            run_benchmark(cases, dataset=self.manifest["dataset"], provider=provider,
                          scorer=lambda *_: 1, scorer_metadata=self.manifest["scorer"], manifest=self.manifest)
        provider.generate.assert_not_called()

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_benchmark.py"), *map(str, args)],
                              cwd=ROOT, text=True, capture_output=True)

    def test_export_and_reference_rerun_reconstruct_private_solutions(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, result_path = Path(directory) / "manifest.json", Path(directory) / "result.json"
            proc = self.cli("--manifest-only", "--export-manifest", manifest_path, "--samples-per-domain", "1", "--difficulty", "1")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(result_path.exists())
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest, self.manifest)
            proc = self.cli("--manifest", manifest_path, "--output", result_path)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(result_path.read_text())
            self.assertEqual(result["dataset"]["manifest_sha256"], manifest["manifest_sha256"])
            self.assertEqual(result["summary"]["sample_count"], 4)
            self.assertEqual(result["summary"]["mean_score"], 1)
            proc = self.cli("--manifest", manifest_path, "--seed", "1", "--output", result_path)
            self.assertEqual(proc.returncode, 2)
            proc = self.cli("--manifest", manifest_path, "--output", manifest_path)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(load_manifest(manifest_path), manifest)

    def test_reference_rejects_rehashed_modified_tasks_and_scorer_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, result_path = Path(directory) / "manifest.json", Path(directory) / "result.json"
            for field in ("prompt", "version"):
                bad = copy.deepcopy(self.manifest)
                if field == "prompt":
                    bad["cases"][0]["input"]["prompt"] = "modified"
                else:
                    bad["scorer"]["version"] = "different-revision"
                bad["manifest_sha256"] = manifest_digest(bad)
                save_manifest(bad, manifest_path)
                proc = self.cli("--manifest", manifest_path, "--output", result_path)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertFalse(result_path.exists())


if __name__ == "__main__":
    unittest.main()
