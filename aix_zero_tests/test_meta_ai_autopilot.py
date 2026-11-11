"""Unit tests for the AIX Meta-AI Autonomous Self-Improvement Engine and kill-switch CLI."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from aix_zero.aix_ctl import main as aix_ctl_main
from aix_zero.meta_ai_autopilot import (
    AutonomousSelfImprovementController,
    META_AI_EXPERT_DOMAINS,
    build_meta_ai_self_improvement_curriculum,
    build_meta_training_batch,
    scan_open_source_ai_codebase_patterns,
    synthesize_peer_ai_improvement_prompt,
)
from aix_zero.tokenizer import build_v11_tokenizer


class MetaAIAutopilotTests(unittest.TestCase):
    def test_meta_ai_curriculum_and_open_source_codebase_index(self):
        curr = build_meta_ai_self_improvement_curriculum()
        domains = {c["domain"] for c in curr}
        self.assertEqual(domains, set(META_AI_EXPERT_DOMAINS.keys()))
        self.assertGreaterEqual(len(curr), 25)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("aix_zero", "rlvr-forge", "branchbox", "mlx-tree", "steerlens"):
                package = root / name
                package.mkdir()
                (package / "example.py").write_text("def example():\n    pass\n", encoding="utf-8")
            idx = scan_open_source_ai_codebase_patterns(root)
        self.assertEqual(idx["indexed_package_count"], 5)
        self.assertEqual(len(idx["cycle_focused_files"]), 3)
        peer = synthesize_peer_ai_improvement_prompt(idx, target_domain="META_RLVR_GRPO_TRAINER")
        self.assertIn("META_RLVR_GRPO_TRAINER", peer["peer_coaching_prompt"])

    def test_training_scores_complete_answers_from_every_domain(self):
        curriculum = build_meta_ai_self_improvement_curriculum(27)
        tokenizer = build_v11_tokenizer()
        ids, labels = build_meta_training_batch(curriculum, tokenizer, max_seq_len=256)

        self.assertEqual(ids.shape[0], len(curriculum))
        domains = set()
        for item, tokens, targets in zip(curriculum, ids, labels):
            prompt = tokenizer.encode(item["prompt"])
            completion = tokenizer.encode(item["completion"], add_bos=False, add_eos=True)
            end = len(prompt) + len(completion)
            self.assertEqual(tokens[:end].tolist(), prompt + completion)
            self.assertEqual(targets[:len(prompt)].tolist(), [0] * len(prompt))
            self.assertEqual(targets[len(prompt):end].tolist(), completion)
            self.assertIn(f"<ans>{item['gold']}</ans>", tokenizer.decode(completion))
            self.assertTrue((targets[end:] == 0).all())
            domains.add(item["domain"])
        self.assertEqual(domains, set(META_AI_EXPERT_DOMAINS))

        with self.assertRaisesRegex(ValueError, "exceeds the model sequence length"):
            build_meta_training_batch(curriculum, tokenizer, max_seq_len=48)

    def test_autopilot_cycle_and_immediate_shutdown_kill_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ctrl = AutonomousSelfImprovementController(tdir, max_cpu_threads=1)

            # Run Cycle 1 and Cycle 2 and verify progressive, distinct work each cycle
            res1 = ctrl.run_single_self_improvement_cycle(cartridge_bank_dir=tdir / "bank")
            res2 = ctrl.run_single_self_improvement_cycle(cartridge_bank_dir=tdir / "bank")
            self.assertTrue(res1["executed"])
            self.assertTrue(res2["executed"])
            self.assertEqual(res1["curriculum_verified_count"], res1["curriculum_total_count"])
            self.assertEqual(res2["curriculum_verified_count"], res2["curriculum_total_count"])
            self.assertNotEqual(res1["curriculum_sha256"], res2["curriculum_sha256"])
            self.assertNotEqual(res1["cycle_focused_files"], res2["cycle_focused_files"])
            self.assertLess(res1["expert_post_update_loss"], res1["expert_pre_update_loss"])
            self.assertLess(res2["expert_post_update_loss"], res2["expert_pre_update_loss"])
            self.assertNotEqual(res1["shard_hash_prefixes"], res2["shard_hash_prefixes"])
            self.assertEqual(len(res2["exported_meta_shards"]), 4)
            self.assertEqual(res2["training_example_count"], res2["curriculum_total_count"])
            self.assertEqual(res2["training_objective"], "completion_token_cross_entropy")

            # Trigger stop via CLI (`aix_ctl stop`)
            rc = aix_ctl_main(["--state-dir", str(tdir), "stop", "--reason", "USER_SHUTTING_DOWN_MAC"])
            self.assertEqual(rc, 0)
            self.assertTrue(ctrl.should_stop())

            # Attempting another cycle while stopped halts immediately without running
            blocked = ctrl.run_single_self_improvement_cycle(cartridge_bank_dir=tdir / "bank")
            self.assertFalse(blocked["executed"])
            self.assertTrue(blocked["halted_by_kill_switch"])
            self.assertEqual(ctrl.get_status()["last_stop_reason"], "USER_SHUTTING_DOWN_MAC")


if __name__ == "__main__":
    unittest.main()
