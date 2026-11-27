"""Unit tests for reproducible adapter export with lineage & round-trip verification (`#443`)."""
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.adapter_export import export_verified_adapter_bundle
from aix_zero.benchmark_baseline import load_v11
from aix_zero.meta_ai_autopilot import REPO_ROOT


class AdapterExportTests(unittest.TestCase):
    def test_atomic_export_roundtrip_and_no_overwrite(self):
        ckpt = REPO_ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        model, _ = load_v11(ckpt)
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            pt_path = tdir / "candidate.pt"
            torch.save(model.state_dict(), pt_path)
            out_dir = tdir / "release-v1"
            manifest = export_verified_adapter_bundle(
                source_path=pt_path,
                output_dir=out_dir,
                base_identity={"checkpoint": ckpt.name, "sha256": "abc"},
                training_recipe={"lr": 1e-3, "steps": 5},
                dataset_provenance={"source": "verified_synthetic", "license": "MIT"},
                evaluation_summary={"neural_only_accuracy": 0.3165},
                release_label="latest_candidate",
                verify_aix_zero_roundtrip_ckpt=ckpt,
            )
            self.assertTrue((out_dir / "export_manifest.json").exists())
            self.assertTrue((out_dir / "artifact" / "candidate.pt").exists())
            self.assertTrue(manifest["roundtrip_verification"]["verified"])
            self.assertEqual(manifest["roundtrip_verification"]["max_logit_abs_diff"], 0.0)
            self.assertEqual(manifest["mergekit_lineage"]["parent_models"], ["unknown"])

            with self.assertRaises(FileExistsError):
                export_verified_adapter_bundle(
                    source_path=pt_path,
                    output_dir=out_dir,
                    base_identity={"checkpoint": ckpt.name},
                    training_recipe={},
                    dataset_provenance={},
                    evaluation_summary={},
                )


if __name__ == "__main__":
    unittest.main()
