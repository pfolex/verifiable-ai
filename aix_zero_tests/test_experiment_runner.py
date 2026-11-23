import hashlib
from pathlib import Path
import tempfile
import unittest
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_splits import build_split_manifest
from aix_zero.experiment_runner import run_experiment


class RunnerTests(unittest.TestCase):
    def test_recorded_rejection_keeps_baseline_and_hides_test_answers(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'model';source.write_bytes(b'champion')
            registry=ExperimentRegistry(Path(d)/'registry')
            rows={s:[dict(prompt=s,gold=s,completion=s)] for s in ('train','development','regression','final')}
            manifest=build_split_manifest(rows,lambda r:r['prompt'])
            metadata=dict(source_revision='revision',tokenizer_sha256='c'*64,dataset_sha256=manifest['sha256'],seed=1,hardware='cpu',settings={})
            policy=dict(min_cases=1,min_net_gain=1,max_regression_losses=0,max_trace_losses=0)
            def train(data,items,config):
                self.assertEqual(data,b'champion');self.assertEqual(items,rows['train'])
                self.assertEqual(config['settings']['policy'],policy)
                return b'candidate'
            def evaluate(manifest,a,b,split):
                record=dict(prompt=split,gold=split,scored=True,answer_correct=False,trace_valid=False)
                arm=dict(complete=True,offered=1,scored=1,records=[record])
                return dict(split=split,comparison_complete=True,assistance_mode='neural_only',settings={},tokenizer_sha256='c'*64,split_manifest_sha256=manifest['sha256'],model_identities=dict(champion=hashlib.sha256(a.read_bytes()).hexdigest(),candidate=hashlib.sha256(b.read_bytes()).hexdigest()),arms=dict(champion=arm,candidate=arm))
            result=run_experiment(registry,source,manifest,metadata,policy,identity=lambda r:r['prompt'],train=train,evaluate=evaluate)
            self.assertEqual(result['state'],'rejected');self.assertEqual(source.read_bytes(),b'champion')
            self.assertEqual(len(registry.inspect(result['attempt'])['events']),4)
            def fail(*args):raise RuntimeError('training failed')
            with self.assertRaises(RuntimeError):
                run_experiment(registry,source,manifest,metadata,policy,identity=lambda r:r['prompt'],train=fail,evaluate=evaluate)
            states=[registry.inspect(p.name)['state'] for p in (registry.root/'attempts').iterdir()]
            self.assertCountEqual(states,['rejected','failed'])

            def improved(manifest,a,b,split):
                report=evaluate(manifest,a,b,split)
                import copy
                report=copy.deepcopy(report)
                report['arms']['candidate']=copy.deepcopy(report['arms']['candidate'])
                if split=='development':
                    report['arms']['candidate']['records'][0]['answer_correct']=True
                return report
            eligible=run_experiment(registry,source,manifest,metadata,policy,identity=lambda r:r['prompt'],train=train,evaluate=improved)
            self.assertEqual(eligible['state'],'evaluated')
            self.assertEqual(eligible['decision']['decision'],'eligible_for_review')
            self.assertEqual(source.read_bytes(),b'champion')
            def wrong_cases(manifest,a,b,split):
                report=improved(manifest,a,b,split)
                for arm in report['arms'].values():arm['records'][0]['prompt']='substituted'
                return report
            with self.assertRaisesRegex(ValueError,'cases differ'):
                run_experiment(registry,source,manifest,metadata,policy,identity=lambda r:r['prompt'],train=train,evaluate=wrong_cases)
            with self.assertRaisesRegex(ValueError,'Invalid policy'):
                run_experiment(registry,source,manifest,metadata,{},identity=lambda r:r['prompt'],train=fail,evaluate=evaluate)
