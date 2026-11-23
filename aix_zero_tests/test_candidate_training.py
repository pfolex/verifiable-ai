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
