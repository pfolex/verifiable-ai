import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from aix_zero.inspect_candidate_experiment import (
    classify_case_failure,
    compare_paired_cases,
    inspect_attempt,
    main as inspect_main,
)
from aix_zero.prepare_candidate_experiment import prepare_experiment_inputs
from aix_zero.run_candidate_experiment import main as run_main


class InspectCandidateExperimentTests(unittest.TestCase):
    def test_failure_taxonomy_covers_eos_budget_empty_malformed_and_invalid_trace(self):
        # 1. Valid answer + valid trace + EOS
        ok_rec = dict(
            prompt='Q:MetaQuant:fp16_bits=16;quant_bits=4->compression?',
            gold='4',
            completion='<think>comp=16/4=4</think><ans>4</ans><eos>',
            scored=True,
            answer_correct=True,
            trace_valid=True,
            termination_reason='eos',
            generated_tokens=15,
        )
        c_ok = classify_case_failure(ok_rec)
        self.assertEqual(c_ok['failure_kind'], 'none')
        self.assertEqual(c_ok['termination_reason'], 'eos')

        # 2. Correct final answer with invalid intermediate calculation trace
        bad_trace_rec = dict(
            prompt='Q:Phys:a=3,t=5,u=2,v=',
            gold='17',
            completion='<think>3*5=99;99+2=17</think><ans>17</ans><eos>',
            scored=True,
            answer_correct=True,
            trace_valid=False,
            termination_reason='eos',
            generated_tokens=16,
        )
        c_bt = classify_case_failure(bad_trace_rec)
        self.assertEqual(c_bt['failure_kind'], 'correct_answer_invalid_trace')
        self.assertEqual(c_bt['trace_status'], 'invalid_intermediate_calculation')

        # 3. Budget exhaustion (max_new_tokens) with unclosed answer tag
        trunc_rec = dict(
            prompt='Q:MetaScale:C=52995600;N=4200->D_opt?',
            gold='2103',
            completion='<think>4=4/12044;55122/540=1/11111=1012/5;',
            scored=True,
            answer_correct=False,
            trace_valid=False,
            termination_reason='max_new_tokens',
            generated_tokens=36,
        )
        c_tr = classify_case_failure(trunc_rec)
        self.assertEqual(c_tr['failure_kind'], 'missing_or_unclosed_answer_tag')
        self.assertEqual(c_tr['termination_reason'], 'max_new_tokens')

        # 4. Empty output
        empty_rec = dict(
            prompt='Q:MetaPyAsync:seq_ms=75;async_ms=25->concurrency_gain?',
            gold='3',
            completion='<eos>',
            scored=True,
            answer_correct=False,
            trace_valid=False,
            termination_reason='eos',
            generated_tokens=1,
        )
        c_emp = classify_case_failure(empty_rec)
        self.assertEqual(c_emp['failure_kind'], 'empty_output')

        # 5. Malformed fraction inside <ans>...</ans>
        mal_rec = dict(
            prompt='Q:MetaMuon:muon_states=1;adam_states=4->state_ratio?',
            gold='1/4',
            completion='<think>1/4=1/4</think><ans>1/0</ans><eos>',
            scored=True,
            answer_correct=False,
            trace_valid=False,
            termination_reason='eos',
            generated_tokens=14,
        )
        c_mal = classify_case_failure(mal_rec)
        self.assertEqual(c_mal['failure_kind'], 'malformed_answer')

        # 6. Incomplete / generation error
        err_rec = dict(
            prompt='Q:Err',
            gold='1',
            scored=False,
            answer_correct=False,
            trace_valid=False,
            error_type='RuntimeError',
            error='interrupted',
        )
        c_err = classify_case_failure(err_rec)
        self.assertEqual(c_err['failure_kind'], 'generation_error')

    def test_same_case_answer_and_trace_loss_and_cli_probe(self):
        # Verify when champion had answer_correct=True, trace_valid=True and candidate lost both on the same case
        report = dict(
            split='development',
            comparison_complete=True,
            settings=dict(max_new_tokens=36),
            context_limit=112,
            paired=dict(gains=0, losses=1, both_correct=0, both_wrong=0),
            arms=dict(
                champion=dict(
                    complete=True,
                    offered=1,
                    scored=1,
                    correct=1,
                    trace_valid=1,
                    records=[dict(
                        prompt='Q:MetaRLOO:r1=1;other_sum=1;k=4->rloo_advantage?',
                        domain='META_TRL_RLOO_LEAVE_ONE_OUT',
                        gold='2/3',
                        completion='<think>base=1/3;adv=1-1/3=2/3</think><ans>2/3</ans><eos>',
                        scored=True,
                        answer_correct=True,
                        trace_valid=True,
                        termination_reason='eos',
                    )],
                ),
                candidate=dict(
                    complete=True,
                    offered=1,
                    scored=1,
                    correct=0,
                    trace_valid=0,
                    records=[dict(
                        prompt='Q:MetaRLOO:r1=1;other_sum=1;k=4->rloo_advantage?',
                        domain='META_TRL_RLOO_LEAVE_ONE_OUT',
                        gold='2/3',
                        completion='<think>ba==1/5=4;=1;p4</ans><eos>',
                        scored=True,
                        answer_correct=False,
                        trace_valid=False,
                        termination_reason='eos',
                    )],
                ),
            ),
        )
        compared = compare_paired_cases(report)
        self.assertEqual(len(compared['answer_losses']), 1)
        self.assertEqual(len(compared['trace_losses']), 1)
        self.assertEqual(compared['answer_losses'][0]['prompt'], compared['trace_losses'][0]['prompt'])

        # Run real attempt and inspect + probe via CLI
        with tempfile.TemporaryDirectory() as tmp:
            bundle = prepare_experiment_inputs(Path(tmp) / 'bundle', steps=1)
            with contextlib.redirect_stdout(io.StringIO()):
                run_res = run_main(bundle['run_argv'][3:])
            out_path = Path(tmp) / 'inspect_report.json'
            with contextlib.redirect_stdout(io.StringIO()):
                inspected = inspect_main([
                    '--registry', bundle['registry_path'],
                    '--attempt', run_res['attempt'],
                    '--probe-max-new-tokens', '48',
                    '--output', str(out_path),
                ])
            self.assertTrue(out_path.is_file())
            self.assertEqual(inspected['attempt'], run_res['attempt'])
            self.assertIn('decoding_budget_probe', inspected)
            self.assertFalse(inspected['decoding_budget_probe']['independent_final_test'])
            self.assertEqual(inspected['decoding_budget_probe']['probed_budget']['max_new_tokens'], 48)


if __name__ == '__main__':
    unittest.main()
