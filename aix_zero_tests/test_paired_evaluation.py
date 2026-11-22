import unittest
from aix_zero.paired_evaluation import evaluate_pair


class PairedTests(unittest.TestCase):
    def run_pair(self, a, b, rows=None):
        return evaluate_pair(rows or [dict(prompt='question',gold='4')],a,b,
                             identities=dict(champion='base',candidate='new'),settings=dict(tokens=36),
                             score_answer=lambda text,gold:text.endswith(gold),
                             score_trace=lambda text,gold:text=='valid:'+gold)

    def test_separate_answer_and_trace_and_prompt_only_generation(self):
        calls=[]
        def generate(prompt, settings):
            calls.append((prompt,dict(settings)));settings['tokens']=1
            return 'invalid:4'
        report=self.run_pair(lambda p,s:'wrong',generate)
        self.assertEqual(calls,[('question',{'tokens':36})])
        self.assertEqual(report['settings'],{'tokens':36})
        self.assertEqual(report['paired']['gains'],1)
        self.assertEqual(report['arms']['candidate']['trace_valid'],0)
        self.assertEqual(report['statistical_status'],'NOT_EVALUATED')
        changed=self.run_pair(generate,generate,[dict(prompt='question',gold='999')])
        self.assertEqual(changed['arms']['candidate']['records'][0]['completion'],'invalid:4')
        self.assertEqual(changed['arms']['candidate']['correct'],0)

    def test_exception_prevents_comparison_and_keeps_failure_record(self):
        def fail(prompt, settings):raise RuntimeError('interrupted')
        report=self.run_pair(lambda p,s:'valid:4',fail)
        self.assertFalse(report['comparison_complete'])
        self.assertIsNone(report['paired'])
        self.assertEqual(report['arms']['candidate']['scored'],0)
        self.assertEqual(report['arms']['candidate']['records'][0]['error_type'],'RuntimeError')

    def test_empty_and_duplicate_sets_fail(self):
        args=dict(identities=dict(champion='a',candidate='b'),settings={},score_answer=lambda t,g:True,score_trace=lambda t,g:True)
        for rows in ([],[dict(prompt='q',gold='1')]*2):
            with self.assertRaises(ValueError):evaluate_pair(rows,lambda p,s:'',lambda p,s:'',**args)
