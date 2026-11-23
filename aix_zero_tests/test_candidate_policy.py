from copy import deepcopy
import unittest
from aix_zero.candidate_policy import screen_candidate


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy=dict(min_cases=1,min_net_gain=1,max_regression_losses=0,max_trace_losses=0)
    def report(self,split,a,b):
        def arm(correct):
            return dict(complete=True,offered=1,scored=1,records=[dict(prompt=split,gold='1',scored=True,answer_correct=correct,trace_valid=correct)])
        return dict(split=split,comparison_complete=True,assistance_mode='neural_only',model_identities=dict(champion='a'*64,candidate='b'*64),settings={},tokenizer_sha256='c'*64,split_manifest_sha256='d'*64,arms=dict(champion=arm(a),candidate=arm(b)))
    def test_improvement_retention_and_memorization_rejection(self):
        reg=self.report('regression',True,True)
        self.assertEqual(screen_candidate(self.report('development',False,True),reg,self.policy)['decision'],'eligible_for_review')
        self.assertEqual(screen_candidate(self.report('development',False,False),reg,self.policy)['decision'],'reject')
        self.assertEqual(screen_candidate(self.report('development',False,True),self.report('regression',True,False),self.policy)['decision'],'reject')
    def test_incomplete_assisted_and_incompatible_reports_block(self):
        dev=self.report('development',False,True);reg=self.report('regression',True,True)
        for key,value in [('comparison_complete',False),('assistance_mode','symbolic'),('split','final')]:
            changed=deepcopy(dev);changed[key]=value
            self.assertEqual(screen_candidate(changed,reg,self.policy)['decision'],'blocked')
        changed=deepcopy(reg);changed['settings']={'different':True}
        self.assertEqual(screen_candidate(dev,changed,self.policy)['decision'],'blocked')
    def test_claimed_summary_cannot_override_per_case_failure(self):
        dev=self.report('development',False,False);dev['paired']={'gains':999}
        self.assertEqual(screen_candidate(dev,self.report('regression',True,True),self.policy)['decision'],'reject')
