"""Mandatory schema and semantic contract checks for evaluation results."""
import copy
import json
import unittest
from pathlib import Path

from jsonschema import ValidationError

from evaluation_result import validate_evaluation_result

ROOT = Path(__file__).resolve().parents[1]


class EvaluationResultContractTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((ROOT / "examples/evaluation-result-v1.json").read_text())

    def test_all_example_artifacts_are_valid(self):
        for path in sorted((ROOT / "examples").glob("evaluation-result-v*.json")):
            with self.subTest(path=path):
                validate_evaluation_result(json.loads(path.read_text()))

    def test_schema_rejects_wrong_version(self):
        self.example["schema_version"] = "2.0.0"
        with self.assertRaises(ValidationError):
            validate_evaluation_result(self.example)

    def test_schema_rejects_missing_required_field(self):
        del self.example["dataset"]
        with self.assertRaises(ValidationError):
            validate_evaluation_result(self.example)

    def test_schema_rejects_invalid_timestamp(self):
        self.example["created_at"] = "not-a-date"
        with self.assertRaises(ValidationError):
            validate_evaluation_result(self.example)

    def test_duplicate_sample_ids_are_rejected(self):
        duplicate = copy.deepcopy(self.example["samples"][0])
        duplicate["output"] = "different output"
        self.example["samples"].append(duplicate)
        self.example["summary"]["sample_count"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            validate_evaluation_result(self.example)

    def test_incorrect_summary_counts_are_rejected(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.example["summary"]["sample_count"] = count
                with self.assertRaisesRegex(ValueError, "sample_count"):
                    validate_evaluation_result(self.example)

    def test_incorrect_mean_scores_are_rejected(self):
        self.example["summary"]["mean_score"] = 0.5
        with self.assertRaisesRegex(ValueError, "mean_score"):
            validate_evaluation_result(self.example)

    def test_multiple_samples_and_rounding(self):
        self.example["samples"] = [
            dict(self.example["samples"][0], sample_id=str(i), score=score)
            for i, score in enumerate((0.1, 0.2, 0.3))
        ]
        self.example["summary"].update(sample_count=3, mean_score=0.2)
        validate_evaluation_result(self.example)
        self.example["summary"]["mean_score"] = 0.3
        with self.assertRaisesRegex(ValueError, "mean_score"):
            validate_evaluation_result(self.example)

    def test_empty_run_has_zero_mean(self):
        self.example["samples"] = []
        self.example["summary"].update(sample_count=0, mean_score=0)
        validate_evaluation_result(self.example)
        self.example["summary"]["mean_score"] = 0.1
        with self.assertRaisesRegex(ValueError, "mean_score"):
            validate_evaluation_result(self.example)

    def test_nonfinite_scores_are_rejected(self):
        self.example["samples"][0]["score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_evaluation_result(self.example)


if __name__ == "__main__":
    unittest.main()
