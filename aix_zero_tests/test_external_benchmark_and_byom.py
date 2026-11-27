"""Unit tests for Pinned External Benchmark Adapter (`#441`) and BYOM Pipeline (`#442`/`#444`)."""
import json
from pathlib import Path
import tempfile
import unittest

from aix_zero.backend_interface import AIXZeroBackendAdapter
from aix_zero.byom_pipeline import run_byom_experiment
from aix_zero.external_benchmark_adapter import (
    evaluate_backend_on_pinned_external_benchmark,
    load_pinned_external_benchmark,
)
from aix_zero.meta_ai_autopilot import REPO_ROOT


class ExternalBenchmarkAndByomTests(unittest.TestCase):
    def test_external_benchmark_rejects_synthetic_substitutes_and_runs_neural_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text(
                json.dumps(
                    {
                        "task_id": "1",
                        "dataset_id": "internal_diag_quant_math",
                        "dataset_revision": "v1",
                        "split": "test",
                        "license": "MIT",
                        "prompt": "1+1=",
                        "gold": "2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "refuses synthetic substitute"):
                load_pinned_external_benchmark(bad)

            good = Path(tmp) / "gsm8k_sample.jsonl"
            good.write_text(
                json.dumps(
                    {
                        "task_id": "gsm_1",
                        "dataset_id": "openai/gsm8k",
                        "dataset_revision": "main@abc123",
                        "split": "test",
                        "license": "MIT",
                        "prompt": "Q:MetaScale:C=144;N=4->D_opt?",
                        "gold": "6",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bundle = load_pinned_external_benchmark(good, expected_dataset_id="openai/gsm8k")
            backend = AIXZeroBackendAdapter(REPO_ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors")
            rep = evaluate_backend_on_pinned_external_benchmark(backend, bundle, max_new_tokens=16)
            self.assertEqual(rep["evaluation_origin"], "pinned_external_benchmark")
            self.assertEqual(rep["assistance_mode"], "neural_only")
            self.assertFalse(rep["generation_settings"]["inflight_alu"])
            self.assertEqual(rep["tested_samples"], 1)

    def test_byom_pipeline_record_reject_only_and_immutable_base(self):
        ckpt = REPO_ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        base_bytes = ckpt.read_bytes()
        rows = [
            {
                "task_id": "tr1",
                "messages": [{"role": "user", "content": "Q:MetaScale:C=144;N=4->D_opt?"}],
                "target": "<think>6*4=24;144/24=6</think><ans>6</ans>",
                "source": "byom_demo",
                "license": "MIT",
                "task_family": "scaling",
                "split": "train",
            },
            {
                "task_id": "dev1",
                "messages": [{"role": "user", "content": "Q:MetaScale:C=288;N=4->D_opt?"}],
                "target": "<think>6*4=24;288/24=12</think><ans>12</ans>",
                "source": "byom_demo",
                "license": "MIT",
                "task_family": "scaling",
                "split": "development",
            },
            {
                "task_id": "ret1",
                "messages": [{"role": "user", "content": "Q:MetaMoE:top_k=2;n_experts=8->active_fraction?"}],
                "target": "<think>active=2/8=1/4</think><ans>1/4</ans>",
                "source": "byom_demo",
                "license": "MIT",
                "task_family": "moe",
                "split": "retention",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ds_path = tdir / "byom.jsonl"
            ds_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            run_dir = tdir / "run1"
            summary = run_byom_experiment(
                model_path=ckpt,
                dataset_path=ds_path,
                run_dir=run_dir,
                steps=2,
                lr=1e-3,
                allow_promotion=False,
            )
            self.assertEqual(summary["decision"], "REJECTED_RECORD_ONLY")
            self.assertTrue(summary["base_preserved"])
            self.assertTrue((run_dir / "byom_summary.json").exists())
            self.assertTrue((run_dir / "exported_bundle" / "export_manifest.json").exists())
            self.assertEqual(ckpt.read_bytes(), base_bytes)


if __name__ == "__main__":
    unittest.main()
