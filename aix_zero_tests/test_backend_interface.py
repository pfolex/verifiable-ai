"""Contract tests for Unified Backend Interface & HF/Mergekit Compatibility Validator (`#438` & `#439`)."""
import json
from pathlib import Path
import tempfile
import unittest

from aix_zero.backend_interface import (
    AIXZeroBackendAdapter,
    HuggingFaceBackendAdapter,
    validate_hf_or_mergekit_model_dir,
)
from aix_zero.meta_ai_autopilot import REPO_ROOT


class BackendInterfaceTests(unittest.TestCase):
    def test_aix_zero_backend_adapter_contract(self):
        ckpt = REPO_ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        adapter = AIXZeroBackendAdapter(ckpt)
        info = adapter.inspect_model()
        self.assertEqual(info["backend"], "aix_zero_native")
        self.assertEqual(info["checkpoint_format"], "aix_zero_safetensors")
        self.assertGreater(info["physical_parameters"], 500000)
        self.assertTrue(info["memory_values_are_estimates"])

        gen = adapter.generate("Q:MetaScale:C=144;N=4->D_opt?", max_new_tokens=16)
        self.assertEqual(gen["assistance_mode"], "neural_only")
        self.assertIn("completion", gen)

        tr = adapter.train_step(
            [{"prompt": "Q:MetaScale:C=144;N=4->D_opt?", "completion": "<think>6*4=24;144/24=6</think><ans>6</ans>"}],
            lr=1e-3,
            steps=2,
        )
        self.assertLessEqual(tr["post_loss"], tr["pre_loss"])

    def test_aix_zero_backend_rejects_gguf_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(b"GGUF")
            with self.assertRaisesRegex(ValueError, "GGUF"):
                AIXZeroBackendAdapter(gguf)

    def test_hf_validator_rejects_gguf_missing_shards_and_unsupported_arch(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # 1. Rejects GGUF directory
            (d / "model.gguf").write_bytes(b"GGUF")
            with self.assertRaisesRegex(ValueError, "Ollama/GGUF"):
                validate_hf_or_mergekit_model_dir(d)
            (d / "model.gguf").unlink()

            # 2. Rejects unsupported architecture
            (d / "config.json").write_text(json.dumps({"model_type": "unsupported_arch"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported model_type"):
                validate_hf_or_mergekit_model_dir(d)

            # 3. Rejects missing shard referenced by index
            (d / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "vocab_size": 1000,
                        "max_position_embeddings": 2048,
                        "num_hidden_layers": 2,
                        "hidden_size": 128,
                        "rope_theta": 10000.0,
                    }
                ),
                encoding="utf-8",
            )
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Missing safetensors shards"):
                validate_hf_or_mergekit_model_dir(d)

    def test_hf_validator_accepts_tiny_compatible_mergekit_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "architectures": ["Qwen2ForCausalLM"],
                        "vocab_size": 1000,
                        "max_position_embeddings": 2048,
                        "num_hidden_layers": 2,
                        "hidden_size": 128,
                        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                    }
                ),
                encoding="utf-8",
            )
            (d / "model.safetensors").write_bytes(b"stub_safetensors_bytes")
            (d / "tokenizer_config.json").write_text(
                json.dumps({"eos_token": "<|endoftext|>", "chat_template": "{{ messages }}"}),
                encoding="utf-8",
            )
            (d / "mergekit_config.yml").write_text(
                "models:\n  - model: Qwen/Qwen2.5-0.5B-Instruct\nmerge_method: ties\n",
                encoding="utf-8",
            )
            adapter = HuggingFaceBackendAdapter(d, requested_device="cpu")
            info = adapter.inspect_model()
            self.assertTrue(info["valid"])
            self.assertEqual(info["compatibility_status"], "TESTED_SUPPORTED")
            self.assertTrue(info["mergekit_lineage"]["is_mergekit_export"])
            self.assertEqual(info["mergekit_lineage"]["parent_models"], ["Qwen/Qwen2.5-0.5B-Instruct"])
            self.assertFalse(info["trust_remote_code"])
            self.assertTrue(info["memory_values_are_estimates"])

            # Verify actionable dependency error when transformers/peft are unavailable
            try:
                adapter.train_step([{"prompt": "hi", "completion": "hello"}], steps=1)
            except RuntimeError as exc:
                self.assertIn("requirements-training.txt", str(exc))


if __name__ == "__main__":
    unittest.main()
