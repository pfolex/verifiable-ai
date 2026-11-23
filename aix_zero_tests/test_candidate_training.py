import unittest
from aix_zero.candidate_training import validate_training_settings


class TrainingSettingsTests(unittest.TestCase):
    def test_invalid_and_unbounded_settings_rejected(self):
        valid = dict(steps=2, learning_rate=.001, max_seconds=30)
        validate_training_settings(valid)
        for key, values in dict(steps=[True, 0, 1001], learning_rate=[float('nan'), 0, .1, True], max_seconds=[float('inf'), 0, 3601]).items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_training_settings(dict(valid, **{key: value}))
        with self.assertRaises(ValueError):
            validate_training_settings(dict(valid, extra=1))

    def test_real_checkpoint_training_and_cooperative_timeout(self):
        import hashlib
        import io
        from pathlib import Path
        import torch
        from aix_zero.benchmark_baseline import load_v11
        from aix_zero.candidate_training import train_checkpoint
        from aix_zero.meta_ai_autopilot import build_meta_ai_self_improvement_curriculum
        base=Path(__file__).resolve().parents[1]/'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer=load_v11(base)
        stream=io.BytesIO();torch.save(model.state_dict(),stream);initial=stream.getvalue()
        original=hashlib.sha256(initial).hexdigest()
        row=build_meta_ai_self_improvement_curriculum(cycle_index=101)[0]
        metadata=dict(seed=42,tokenizer_sha256=tokenizer.vocab_sha256,
                      settings=dict(training=dict(steps=1,learning_rate=.0004,max_seconds=60)))
        output=train_checkpoint(initial,[row],metadata)
        self.assertEqual(hashlib.sha256(initial).hexdigest(),original)
        updated=torch.load(io.BytesIO(output.checkpoint),weights_only=True)
        self.assertTrue(any(not torch.equal(t,updated[k]) for k,t in model.state_dict().items()))
        self.assertEqual(output.metrics['steps_completed'],1)
        self.assertEqual(output.metrics['encoding']['encoded'],1)
        metadata['settings']['training']['max_seconds']=1e-12
        with self.assertRaises(TimeoutError):train_checkpoint(initial,[row],metadata)

    def test_bad_trace_rejected(self):
        from aix_zero.candidate_training import train_checkpoint
        metadata=dict(settings=dict(training=dict(steps=1,learning_rate=.0004,max_seconds=60)))
        with self.assertRaisesRegex(ValueError,'trace verification'):
            train_checkpoint(b'not loaded',[dict(prompt='1+1?',gold='2',completion='wrong')],metadata)

    def test_experiment_cli_records_real_training_and_paired_evidence(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        import torch
        from aix_zero.benchmark_baseline import load_v11
        from aix_zero.experiment_registry import ExperimentRegistry
        from aix_zero.experiment_splits import build_split_manifest
        from aix_zero.meta_ai_autopilot import build_meta_ai_self_improvement_curriculum
        from aix_zero.run_candidate_experiment import main
        base=Path(__file__).resolve().parents[1]/'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer=load_v11(base)
        splits={name:[] for name in ('train','development','regression','final')}
        seen=set()
        for split, cycle in (('train',101),('development',2100),('regression',3100)):
            for row in build_meta_ai_self_improvement_curriculum(cycle_index=cycle):
                length=len(tokenizer.encode(row['prompt'],add_bos=True,add_eos=False)+
                           tokenizer.encode(row['completion'],add_bos=False,add_eos=True))
                if row['prompt'] not in seen and length <= model.config.max_seq_len:
                    splits[split].append(row);seen.add(row['prompt']);break
            self.assertEqual(len(splits[split]),1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            champion=root/'champion.pt'
            torch.save(model.state_dict(),champion)
            before=champion.read_bytes()
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(build_split_manifest(splits,lambda row:row['prompt'])))
            policy=root/'policy.json'
            policy.write_text(json.dumps(dict(min_cases=1,min_net_gain=1,max_regression_losses=0,max_trace_losses=0)))
            with contextlib.redirect_stdout(io.StringIO()):
                result=main(['--champion',str(champion),'--manifest',str(manifest),
                    '--policy',str(policy),'--registry',str(root/'registry'),
                    '--steps','1','--learning-rate','0.0004','--max-training-seconds','60'])
            self.assertIn(result['state'],('evaluated','rejected'))
            self.assertEqual(champion.read_bytes(),before)
            registry=ExperimentRegistry(root/'registry')
            self.assertEqual(registry.inspect(result['attempt'])['state'],result['state'])
            metrics=json.loads(registry.verify(result['training_metrics_sha256']).read_text())
            self.assertEqual(metrics['steps_completed'],1)
            evidence=json.loads((root/'registry'/'attempts'/result['attempt']/'evaluation.json').read_text())
            for split in ('development','regression'):
                report=evidence[split]
                self.assertTrue(report['comparison_complete'])
                self.assertEqual(report['assistance_mode'],'neural_only')
                self.assertEqual(report['arms']['candidate']['offered'],1)
