"""Schema contract checks; jsonschema is optional for full Draft 2020-12 validation."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EvaluationResultContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "schemas/evaluation-result-v1.schema.json").read_text())
        cls.example = json.loads((ROOT / "examples/evaluation-result-v1.json").read_text())

    def test_schema_is_versioned(self):
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], "1.0.0")

    def test_example_has_required_fields(self):
        self.assertFalse(set(self.schema["required"]) - set(self.example))

    def test_sample_count_and_mean_match(self):
        samples = self.example["samples"]
        self.assertEqual(len(samples), self.example["summary"]["sample_count"])
        self.assertAlmostEqual(sum(s["score"] for s in samples) / len(samples), self.example["summary"]["mean_score"])

    def test_json_schema_validation_when_dependency_installed(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("Install jsonschema for full JSON Schema validation")
        Draft202012Validator.check_schema(self.schema)
        Draft202012Validator(self.schema).validate(self.example)
        bad = dict(self.example, schema_version="2.0.0")
        with self.assertRaises(Exception):
            Draft202012Validator(self.schema).validate(bad)


if __name__ == "__main__":
    unittest.main()
