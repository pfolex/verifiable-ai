"""Verify the minimal CPU research environment lock and offline smoke execution (BASE-02)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest

import jsonschema
import numpy
import safetensors
import torch

from aix_zero.benchmark_baseline import load_v11
from aix_zero.train_from_scratch import extract_answer

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentLockTests(unittest.TestCase):
    def test_requirements_aix_zero_pins_core_packages(self):
        req_path = ROOT / "requirements-aix-zero.txt"
        self.assertTrue(req_path.exists())
        lines = [
            line.strip()
            for line in req_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("torch==2.2.2", lines)
        self.assertIn("numpy==1.26.4", lines)
        self.assertIn("safetensors==0.8.0", lines)
        self.assertIn("-r requirements-test.txt", lines)

    def test_runtime_packages_are_importable_on_cpu(self):
        self.assertTrue(torch.__version__.startswith("2.2.2"))
        self.assertTrue(numpy.__version__.startswith("1.26."))
        self.assertTrue(safetensors.__version__.startswith("0.8."))
        self.assertTrue(hasattr(jsonschema, "Draft202012Validator"))

    def test_offline_cpu_smoke_inference_does_not_mutate_v11_checkpoint(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        before_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        model, tokenizer = load_v11(ckpt)
        prompt = "Q:(9+7)%3="
        ids, conf, ponder = model.generate_verified(
            tokenizer.encode(prompt),
            eos_id=tokenizer.eos_id,
            max_new_tokens=24,
            num_candidates=1,
            temperature=0.0,
            tokenizer=tokenizer,
            inflight_alu=True,
        )
        trace = tokenizer.decode(ids, skip_special=True)
        self.assertEqual(extract_answer(trace), "1")
        self.assertGreaterEqual(conf, 0.0)
        self.assertGreaterEqual(ponder, 1.0)
        after_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        self.assertEqual(before_sha, after_sha)


if __name__ == "__main__":
    unittest.main()
