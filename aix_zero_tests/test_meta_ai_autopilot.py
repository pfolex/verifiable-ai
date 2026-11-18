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
    scan_open_source_ai_codebase_patterns,
    synthesize_peer_ai_improvement_prompt,
)


class MetaAIAutopilotTests(unittest.TestCase):
    def test_meta_ai_curriculum_and_open_source_codebase_index(self):
        curr = build_meta_ai_self_improvement_curriculum()
        domains = {c["domain"] for c in curr}
        self.assertEqual(domains, set(META_AI_EXPERT_DOMAINS.keys()))
        self.assertGreaterEqual(len(curr), 25)

        idx = scan_open_source_ai_codebase_patterns()
        self.assertGreaterEqual(idx["indexed_package_count"], 5)
        peer = synthesize_peer_ai_improvement_prompt(idx, target_domain="META_RLVR_GRPO_TRAINER")
        self.assertIn("META_RLVR_GRPO_TRAINER", peer["peer_coaching_prompt"])

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
            self.assertLessEqual(res1["heldout_val_post_loss"], res1["heldout_val_pre_loss"])
            self.assertLessEqual(res2["heldout_val_post_loss"], res2["heldout_val_pre_loss"])
            self.assertTrue(res1["promoted_on_heldout_val"])
            self.assertTrue(res2["promoted_on_heldout_val"])
            self.assertEqual(res1["code_synthesis_verified_count"], res1["code_synthesis_total_count"])
            self.assertEqual(res2["code_synthesis_verified_count"], res2["code_synthesis_total_count"])
            self.assertEqual(res1["alu_assisted_heldout_exact_match"], 1.0)
            self.assertEqual(res2["alu_assisted_heldout_exact_match"], 1.0)
            self.assertNotEqual(res1["shard_hash_prefixes"], res2["shard_hash_prefixes"])
            self.assertEqual(len(res2["exported_meta_shards"]), 4)

            # Verify interactive query and peer-AI coaching CLI
            q_out = ctrl.query_evolving_meta_ai("Q:MetaScale:C=144;N=4->D_opt?", use_alu=True)
            self.assertEqual(q_out["routed_domain"], "META_SCALING_OPTIMIZER")
            self.assertIn("<ans>6</ans>", q_out["final_completion"])

            c_bad = ctrl.coach_peer_ai_response(
                "Q:MetaRLVR:r=1;successes=2;group=8->grpo_advantage?",
                "<think>mean=2/8;adv=1-2/8=5/8</think><ans>5/8</ans>",
            )
            self.assertFalse(c_bad["peer_correct"])
            self.assertEqual(c_bad["reference_gold"], "3/4")
            self.assertIn("CORRECTION REQUIRED", c_bad["coaching_feedback"])

            c_good = ctrl.coach_peer_ai_response(
                "Q:MetaRLVR:r=1;successes=2;group=8->grpo_advantage?",
                "<think>mean=2/8;adv=1-1/4=3/4</think><ans>3/4</ans>",
            )
            self.assertTrue(c_good["peer_correct"])

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
