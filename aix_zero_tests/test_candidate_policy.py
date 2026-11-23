from copy import deepcopy
import unittest
from aix_zero.candidate_policy import compute_retention_metrics, screen_candidate


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy=dict(min_cases=1,min_net_gain=1,max_regression_losses=0,max_trace_losses=0)
    def report(self,split,a,b):
        def arm(correct):
            return dict(complete=True,offered=1,scored=1,records=[dict(prompt=split,gold='1',scored=True,answer_correct=correct,trace_valid=correct)])
        return dict(split=split,comparison_complete=True,assistance_mode='neural_only',model_identities=dict(champion='a'*64,candidate='b'*64),settings={},tokenizer_sha256='c'*64,split_manifest_sha256='d'*64,arms=dict(champion=arm(a),candidate=arm(b)))
    def multi_report(self,split,pairs):
        champ_records=[];cand_records=[]
        for idx,(a_ans,a_tr,b_ans,b_tr) in enumerate(pairs):
            prompt=f'{split}-q{idx}'
            champ_records.append(dict(prompt=prompt,gold='1',scored=True,answer_correct=a_ans,trace_valid=a_tr))
            cand_records.append(dict(prompt=prompt,gold='1',scored=True,answer_correct=b_ans,trace_valid=b_tr))
        def arm(rows):
            return dict(complete=True,offered=len(rows),scored=len(rows),
                        correct=sum(r['answer_correct'] for r in rows),
                        trace_valid=sum(r['trace_valid'] for r in rows),records=rows)
        return dict(split=split,comparison_complete=True,assistance_mode='neural_only',
                    model_identities=dict(champion='a'*64,candidate='b'*64),settings={},
                    tokenizer_sha256='c'*64,split_manifest_sha256='d'*64,
                    arms=dict(champion=arm(champ_records),candidate=arm(cand_records)))
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
    def test_retention_all_wrong_all_correct_partial_and_skill_tradeoff(self):
        # 1. All-wrong 0/18 case: zero losses is non-informative and retention is not measurable
        all_wrong=self.multi_report('regression',[(False,False,False,False)]*18)
        m_wrong=compute_retention_metrics(all_wrong,min_champion_correct=1)
        self.assertEqual(m_wrong['champion_correct'],0)
        self.assertEqual(m_wrong['candidate_correct'],0)
        self.assertEqual(m_wrong['retained_champion_correct'],0)
        self.assertEqual(m_wrong['losses'],0)
        self.assertEqual(m_wrong['retention_denominator'],0)
        self.assertFalse(m_wrong['retention_measurable'])
        self.assertIsNone(m_wrong['retention_rate'])
        self.assertTrue(m_wrong['zero_losses_non_informative'])
        self.assertEqual(m_wrong['retention_status'],'not_measurable_zero_champion_successes')
        self.assertFalse(m_wrong['champion_coverage_sufficient'])

        # Policy with min_champion_regression_correct=1 returns inconclusive when dev gains +1 but reg is 0/18
        dev_gain=self.multi_report('development',[(False,False,True,True)]+[(False,False,False,False)]*17)
        strict_policy=dict(min_cases=18,min_net_gain=1,max_regression_losses=0,max_trace_losses=0,min_champion_regression_correct=1)
        res_inconc=screen_candidate(dev_gain,all_wrong,strict_policy)
        self.assertEqual(res_inconc['decision'],'inconclusive_insufficient_retention_evidence')

        # 2. All-correct case
        all_correct=self.multi_report('regression',[(True,True,True,True)]*4)
        m_corr=compute_retention_metrics(all_correct,min_champion_correct=2)
        self.assertEqual(m_corr['retention_denominator'],4)
        self.assertEqual(m_corr['retained_champion_correct'],4)
        self.assertEqual(m_corr['retention_rate'],1.0)
        self.assertEqual(m_corr['retention_status'],'retained')

        # 3. Partial failure / insufficient coverage
        partial_cov=self.multi_report('regression',[(True,True,True,True),(False,False,False,False)])
        m_part=compute_retention_metrics(partial_cov,min_champion_correct=2)
        self.assertEqual(m_part['retention_status'],'insufficient_champion_coverage')

        # 4. Candidate gains target skill on development (+2) but loses an established skill on regression (-1)
        dev_plus2=self.multi_report('development',[(False,False,True,True),(False,False,True,True)])
        reg_loss1=self.multi_report('regression',[(True,True,False,False),(True,True,True,True)])
        m_tradeoff=compute_retention_metrics(reg_loss1,min_champion_correct=1)
        self.assertEqual(m_tradeoff['retention_rate'],0.5)
        self.assertEqual(m_tradeoff['losses'],1)
        self.assertEqual(m_tradeoff['retention_status'],'regressed')
        tradeoff_decision=screen_candidate(dev_plus2,reg_loss1,dict(min_cases=2,min_net_gain=1,max_regression_losses=0,max_trace_losses=0,min_champion_regression_correct=1))
        self.assertEqual(tradeoff_decision['decision'],'reject')
        self.assertIn('regression losses exceed limit',tradeoff_decision['reasons'])

        # 5. Zero-case report fails closed
        empty_reg=deepcopy(all_correct)
        empty_reg['arms']['champion']['records']=[]
        empty_reg['arms']['candidate']['records']=[]
        with self.assertRaises(ValueError):
            compute_retention_metrics(empty_reg)

