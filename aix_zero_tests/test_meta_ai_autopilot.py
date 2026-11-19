"""Unit tests for the AIX Meta-AI Autonomous Self-Improvement Engine and kill-switch CLI."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from aix_zero.aix_ctl import main as aix_ctl_main
from aix_zero.meta_ai_autopilot import (
    AutonomousSelfImprovementController,
    ElasticLoRAAdapterBank,
    META_AI_EXPERT_DOMAINS,
    analyze_python_code_quality_and_ast,
    build_meta_ai_self_improvement_curriculum,
    detect_python_antipatterns_ast,
    grade_and_merge_python_patches,
    refactor_and_verify_python_code,
    reflect_and_repair_rollout,
    scan_open_source_ai_codebase_patterns,
    solve_python_repo_task_and_emit_diff,
    synthesize_peer_ai_improvement_prompt,
    verify_r1_reasoning_trace,
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
            self.assertEqual(res1["reflected_heldout_exact_match"], 1.0)
            self.assertEqual(res2["reflected_heldout_exact_match"], 1.0)
            self.assertGreater(res1["grpo_positive_trajectories"], 0)
            self.assertGreater(res2["replay_buffer_sampled_count"], 0)
            self.assertNotEqual(res1["shard_hash_prefixes"], res2["shard_hash_prefixes"])
            self.assertEqual(len(res2["exported_meta_shards"]), 20)
            self.assertEqual(res1["swe_patch_verified_count"], res1["swe_patch_total_count"])
            self.assertTrue(res1["python_refactor_verified"])
            self.assertTrue(res1["repo_task_execute_routed_request_verified"])
            self.assertGreaterEqual(res1["arena_overall_accuracy"], 0.0)
            self.assertLessEqual(res1["arena_overall_accuracy"], 1.0)
            self.assertEqual(res1["arena_sprt_status"], "NOT_EVALUATED")
            self.assertEqual(res1["arena_evaluation_mode"], "neural_only")
            self.assertEqual(res1["reflection_assistance_mode"], "symbolic_solver")
            self.assertGreater(res1["streamed_hf_verified_count"], 0)
            self.assertIn("resolved_backlog_item", res1)
            self.assertTrue((tdir / "codebase_knowledge_graph.json").exists())
            self.assertTrue((tdir / "external_open_source_harvest.json").exists())
            self.assertTrue((tdir / "harvested_open_source_curriculum.json").exists())
            self.assertTrue((tdir / "streamed_hf_verified_batch.json").exists())
            self.assertTrue((tdir / "elastic_capacity_ledger.json").exists())
            self.assertTrue((tdir / "episodic_verified_memory.json").exists())
            self.assertTrue((tdir / "frontier_arena_scorecard.json").exists())
            self.assertTrue((tdir / "peer_ai_coaching_pack.json").exists())
            self.assertTrue((tdir / "autopilot_live.json").exists())

            # Verify Master Python AST/bytecode analyzer, anti-pattern detector, refactorer, and repo-task solver
            py_qual = analyze_python_code_quality_and_ast("def add(a: int, b: int) -> int:\n    return a + b\n")
            self.assertTrue(py_qual["syntax_valid"])
            self.assertEqual(py_qual["type_annotation_ratio"], 1.0)
            anti = detect_python_antipatterns_ast("def bad(x=[]):\n    try:\n        return 1/0\n    except:\n        return 0\n")
            self.assertEqual(anti["antipattern_count"], 3)
            ref_res = refactor_and_verify_python_code()
            self.assertEqual(ref_res["post_refactor_antipatterns"], 0)
            self.assertTrue(ref_res["unit_tests_passed"])
            rt_res = solve_python_repo_task_and_emit_diff()
            self.assertTrue(rt_res["all_contract_checks_passed"])
            gm_res = grade_and_merge_python_patches()
            self.assertTrue(gm_res["collaborative_merged_patch_passed"])

            # Verify ElasticLoRAAdapterBank dynamic rank expansion (`GROWTH-01`)
            eb = ElasticLoRAAdapterBank(d_model=128, d_ff=128, initial_rank=4)
            e1 = eb.maybe_expand_rank(True, cycle_index=1)
            self.assertTrue(e1["expanded"])
            self.assertEqual(e1["current_rank"], 8)
            e2 = eb.maybe_expand_rank(True, cycle_index=2)
            self.assertTrue(e2["expanded"])
            self.assertEqual(e2["current_rank"], 16)

            # Verify ElasticBrain plateau detection (`ELASTIC-01`)
            plat_yes = ctrl.detect_domain_plateaus(window=4, min_improvement=0.05, loss_series_override=[6.10, 6.09, 6.08, 6.08])
            self.assertTrue(plat_yes["plateau_detected"])
            plat_no = ctrl.detect_domain_plateaus(window=4, min_improvement=0.05, loss_series_override=[7.20, 6.80, 6.40, 6.00])
            self.assertFalse(plat_no["plateau_detected"])

            # Verify DeepSeek-R1 multi-stage trace verifier and reflection repair (`R1VERIFY-01..02`)
            bad_eq = verify_r1_reasoning_trace("<think>6*4=99;144/24=6</think><ans>6</ans>", "6")
            self.assertFalse(bad_eq["equations_ok"])
            self.assertFalse(bad_eq["fully_verified"])
            rep = reflect_and_repair_rollout("Q:MetaScale:C=144;N=4->D_opt?", "<think>6*4=99</think><ans>99</ans>", "6")
            self.assertTrue(rep["reflection_triggered"])
            self.assertTrue(rep["final_verified"])

            # Verify Peer-AI Symbiosis Bridge & mutual distillation ingestion (`BRIDGE-01..02`)
            peer_sess = ctrl.run_peer_ai_symbiosis_session(max_prompts=4)
            self.assertEqual(peer_sess["peer_post_coaching_accuracy"], 1.0)
            self.assertGreater(peer_sess["verified_peer_traces_ingested"], 0)

            # Verify dry-run auto-push command construction (`AUTOGH-03`)
            push_plan = ctrl.auto_push_verified_state("chore: test push", dry_run=True)
            self.assertTrue(push_plan["dry_run"])
            self.assertEqual(len(push_plan["commands"]), 3)

            # Verify all 16 Phase-II, Phase-III, Phase-IV & Phase-V domains via symbolic query
            for prompt_str, expected_dom, expected_ans in [
                ("Q:MetaFlash:seq_len=2048;tile_block=128->io_reduction?", "META_FLASH_ATTENTION_TILING", "<ans>16</ans>"),
                ("Q:MetaSpec:accepted=3;window=4->accept_ratio?", "META_SPECULATIVE_DECODING", "<ans>3/4</ans>"),
                ("Q:MetaQuant:fp16_bits=16;quant_bits=4->compression?", "META_QUANTIZATION_FP8_INT4", "<ans>4</ans>"),
                ("Q:MetaDist:stages=4;microbatches=12->bubble_fraction?", "META_DISTRIBUTED_TP_PP_DP", "<ans>1/5</ans>"),
                ("Q:MetaMLA:latent=512;n_heads=32;head_dim=64->mla_ratio?", "META_DEEPSEEK_MLA_LATENT_KV", "<ans>1/8</ans>"),
                ("Q:MetaPaged:tokens=50;blocks=4;block_size=16->utilization?", "META_VLLM_PAGED_ATTENTION", "<ans>25/32</ans>"),
                ("Q:MetaRLOO:r1=1;other_sum=1;k=4->rloo_advantage?", "META_TRL_RLOO_LEAVE_ONE_OUT", "<ans>2/3</ans>"),
                ("Q:MetaMerge:keep_num=3;keep_den=4;topk_num=1;topk_den=5->effective_density?", "META_MERGEKIT_TIES_DARE", "<ans>3/20</ans>"),
                ("Q:MetaMuon:muon_states=1;adam_states=2->state_ratio?", "META_MUON_ORTHOGONAL_OPTIMIZER", "<ans>1/2</ans>"),
                ("Q:MetaLenR1:acc=1;used_toks=200;max_toks=1000->len_reward?", "META_OPENR1_LENGTH_PENALTY_REWARD", "<ans>4/5</ans>"),
                ("Q:MetaDSPy:passed_demos=7;total_demos=8->mipro_score?", "META_DSPY_MIPRO_PROMPT_OPT", "<ans>7/8</ans>"),
                ("Q:MetaSWE:passed_tests=15;total_tests=16->patch_pass_rate?", "META_SWE_PATCH_TEST_COVERAGE", "<ans>15/16</ans>"),
                ("Q:MetaPyAST:branches=5;functions=2->cyclomatic_ratio?", "META_PYTHON_AST_COMPLEXITY", "<ans>3</ans>"),
                ("Q:MetaPyAlgo:hash_ops=8;scan_ops=64->speedup_ratio?", "META_PYTHON_ALGORITHMIC_BIG_O", "<ans>8</ans>"),
                ("Q:MetaPyTorch:ckpt_layers=4;total_layers=16->mem_ratio?", "META_PYTORCH_TENSOR_MEMORY_STRIDES", "<ans>1/4</ans>"),
                ("Q:MetaPyAsync:seq_ms=120;async_ms=30->concurrency_gain?", "META_PYTHON_TYPE_CONCURRENCY_ASYNC", "<ans>4</ans>"),
            ]:
                q_dom = ctrl.query_evolving_meta_ai(prompt_str, use_alu=True)
                self.assertEqual(q_dom["routed_domain"], expected_dom)
                self.assertIn(expected_ans, q_dom["final_completion"])

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
