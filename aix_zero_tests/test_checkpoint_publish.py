"""Checkpoint integrity, real model/cartridge reload, boundary races and publication gates."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

from aix_zero.checkpoint_control import (
    PAUSE, CYCLE_LOCK, PUBLISH_LOCK, atomic_json, cycle_boundary, file_lock, source_fingerprint,
)
from aix_zero.checkpoint_publish import (
    create_snapshot, publish_release, resume_paused, verify_bundle, digest,
)


class CheckpointPublishTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name);self.repo=self.root/'repo';self.repo.mkdir()
        self.state=self.repo/'aix_zero/autopilot_state';self.state.mkdir(parents=True)
        self.out=self.root/'releases'
        self.baseline=self.repo/'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        self.baseline.parent.mkdir(parents=True);self.baseline.write_bytes(b'baseline')
        (self.repo/'main.py').write_text('print("fixture")\n')
        for args in (['init','-q'],['add','main.py'],
                     ['-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture']):
            subprocess.run(['git','-C',str(self.repo),*args],check=True,capture_output=True)
        atomic_json(self.state/'autopilot_status.json',dict(daemon_state='STOPPED',cycles_completed=7))
        (self.state/'aix_zero_v12_evolving.pt').write_bytes(b'model')
        (self.state/'episodic_verified_memory.json').write_text('{"entries": {}}')
        self.info=dict(model_config={'fixture':True},tokenizer={'vocab_sha256':'a'*64},
                       runtime_baseline_sha256=digest(self.baseline),checkpoint_format='fixture_only',
                       physical_accounting={})
        self.model_patch=patch('aix_zero.checkpoint_publish.model_info',return_value=self.info)
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop();self.temporary.cleanup()

    def test_verified_bundle_latest_is_not_best_and_tamper_rejected(self):
        self.out.mkdir();best=self.out/'best.json';best.write_text('unchanged')
        snapshot=create_snapshot(self.state,self.repo,self.out)
        manifest=verify_bundle(snapshot['bundle'])
        self.assertFalse(manifest['independently_evaluated_best'])
        self.assertEqual(manifest['cycle'],7)
        self.assertIn('state/episodic_verified_memory.json',manifest['files'])
        self.assertEqual(best.read_text(),'unchanged')
        self.assertTrue((self.state/PAUSE).exists())
        self.assertEqual(json.loads((self.out/'latest.json').read_text())['tag'],snapshot['tag'])
        with (Path(snapshot['bundle'])/'state/aix_zero_v12_evolving.pt').open('ab') as stream:stream.write(b'bad')
        with self.assertRaisesRegex(ValueError,'integrity'): verify_bundle(snapshot['bundle'])

    def test_dirty_code_byte_cap_and_untracked_code_fail_paused(self):
        (self.repo/'main.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'committed'):create_snapshot(self.state,self.repo,self.out)
        self.assertTrue((self.state/PAUSE).exists())
        (self.repo/'main.py').write_text('print("fixture")\n')
        (self.repo/'new.py').write_text('pass')
        with self.assertRaisesRegex(RuntimeError,'Untracked'):create_snapshot(self.state,self.repo,self.out)
        (self.repo/'new.py').unlink()
        with self.assertRaisesRegex(ValueError,'byte budget'):create_snapshot(self.state,self.repo,self.out,max_bytes=1)
        self.assertFalse((self.out/'latest.json').exists())

    def test_old_active_daemon_and_incomplete_cycle_fail_closed(self):
        (self.state/'autopilot.pid').write_text(str(os.getpid()))
        with self.assertRaisesRegex(RuntimeError,'protocol'):create_snapshot(self.state,self.repo,self.out)
        self.assertFalse((self.state/PAUSE).exists())
        (self.state/'autopilot.pid').unlink()
        for phase in ('failed','active','interrupted'):
            atomic_json(self.state/'autopilot_status.json',dict(daemon_state='STOPPED',cycles_completed=7,checkpoint_cycle_phase=phase))
            with self.assertRaisesRegex(RuntimeError,'incomplete'):create_snapshot(self.state,self.repo,self.out)

    def test_duplicate_publisher_and_cycle_timeout_preserve_pause(self):
        with file_lock(self.state/PUBLISH_LOCK):
            with self.assertRaises(TimeoutError):create_snapshot(self.state,self.repo,self.out)
        with file_lock(self.state/CYCLE_LOCK):
            with self.assertRaises(TimeoutError):create_snapshot(self.state,self.repo,self.out,timeout=.01)
        self.assertTrue((self.state/PAUSE).exists())

    def test_symlink_and_missing_referenced_cartridge_rejected(self):
        extra=self.state/'adapter.pt';extra.symlink_to(self.state/'aix_zero_v12_evolving.pt')
        with self.assertRaises(ValueError):create_snapshot(self.state,self.repo,self.out)
        extra.unlink()
        bank=self.state/'meta_ai_cartridges';bank.mkdir()
        atomic_json(bank/'cartridge_manifest.json',dict(schema_version=1,cartridges={
            'A':dict(cartridge_file='missing.safetensors',sha256='a'*64,serialized_bytes=10)}))
        with self.assertRaises(FileNotFoundError):create_snapshot(self.state,self.repo,self.out)

    def test_source_mutation_during_copy_aborts(self):
        original=shutil.copyfile
        def mutate(source,dest,*args,**kwargs):
            result=original(source,dest,*args,**kwargs)
            if Path(source).name=='aix_zero_v12_evolving.pt':Path(source).write_bytes(b'changed')
            return result
        with patch('aix_zero.checkpoint_publish.shutil.copyfile',side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError,'changed during'):create_snapshot(self.state,self.repo,self.out)
        self.assertTrue((self.state/PAUSE).exists())
        self.assertFalse((self.out/'latest.json').exists())

    def test_boundary_waits_for_completed_cycle_then_blocks_next(self):
        entered=threading.Event();finish=threading.Event();errors=[]
        outer=self
        class Controller:
            state_dir=outer.state;repo_root=outer.repo;status_path=outer.state/'autopilot_status.json'
            def get_status(self):return json.loads(self.status_path.read_text())
            def should_stop(self):return False
            @cycle_boundary
            def run_single_self_improvement_cycle(self):
                entered.set();finish.wait(5)
                (self.state_dir/'aix_zero_v12_evolving.pt').write_bytes(b'completed')
                atomic_json(self.status_path,dict(daemon_state='RUNNING_IDLE_BETWEEN_CYCLES',cycles_completed=8))
                return dict(executed=True)
        controller=Controller()
        def worker():
            try:controller.run_single_self_improvement_cycle()
            except BaseException as e:errors.append(e)
        thread=threading.Thread(target=worker);thread.start()
        self.assertTrue(entered.wait(5))
        with self.assertRaises(TimeoutError):create_snapshot(self.state,self.repo,self.out,timeout=.01)
        finish.set();thread.join(5);self.assertFalse(thread.is_alive());self.assertFalse(errors)
        snapshot=create_snapshot(self.state,self.repo,self.out)
        self.assertEqual(snapshot['cycle'],8)
        self.assertEqual((Path(snapshot['bundle'])/'state/aix_zero_v12_evolving.pt').read_bytes(),b'completed')
        self.assertFalse(controller.run_single_self_improvement_cycle()['executed'])

    def test_runtime_code_change_since_start_rejected(self):
        atomic_json(self.state/'autopilot_status.json',dict(daemon_state='STOPPED',cycles_completed=7,
            checkpoint_protocol_version=1,checkpoint_cycle_phase='idle',checkpoint_runtime_source_sha256='0'*64))
        with self.assertRaisesRegex(RuntimeError,'source changed'):create_snapshot(self.state,self.repo,self.out)

    def fake_gh(self,snapshot,corrupt=False):
        calls=[]
        def runner(args,**kwargs):
            calls.append(args)
            if args[1:3]==['release','download']:
                target=Path(args[args.index('--dir')+1])
                shutil.copyfile(snapshot['archive'],target/Path(snapshot['archive']).name)
                shutil.copyfile(Path(snapshot['archive']).parent/'checksums.json',target/'checksums.json')
                if corrupt:(target/Path(snapshot['archive']).name).write_bytes(b'bad download')
            return subprocess.CompletedProcess(args,0,stdout='',stderr='')
        return runner,calls

    def test_publication_uploads_draft_verifies_download_then_publishes(self):
        snapshot=create_snapshot(self.state,self.repo,self.out)
        runner,calls=self.fake_gh(snapshot)
        result=publish_release(snapshot,'example/aix',runner=runner)
        self.assertTrue(result['verified_download'])
        self.assertIn('--draft',calls[0])
        self.assertEqual(calls[-1][1:3],['release','edit'])
        self.assertIn('--draft=false',calls[-1])
        self.assertTrue((self.out/'latest_published.json').exists())
        self.assertTrue((self.state/PAUSE).exists())

    def test_failed_download_never_publishes_or_updates_published_pointer(self):
        snapshot=create_snapshot(self.state,self.repo,self.out)
        runner,calls=self.fake_gh(snapshot,corrupt=True)
        with self.assertRaisesRegex(ValueError,'Downloaded'):publish_release(snapshot,'example/aix',runner=runner)
        self.assertFalse(any(args[1:3]==['release','edit'] for args in calls))
        self.assertFalse((self.out/'latest_published.json').exists())
        self.assertTrue((self.state/PAUSE).exists())

    def test_resume_remaining_budget_failure_and_clear_only(self):
        create_snapshot(self.state,self.repo,self.out)
        atomic_json(self.state/'.checkpoint.resume.json',dict(was_running=True,pid=99999999,
            launch=dict(max_cycles=5,initial_cycles_completed=6,interval_seconds=2.,threads=1,sync_github=False,auto_push=False)))
        runner=Mock()
        with patch('aix_zero.checkpoint_publish.alive',return_value=False):
            result=resume_paused(self.state,self.repo,runner=runner)
        self.assertEqual(result['remaining_cycles'],4)
        self.assertEqual(runner.call_args.args[0][runner.call_args.args[0].index('--max-cycles')+1],'4')
        atomic_json(self.state/PAUSE,dict(reason='retry'))
        runner.side_effect=RuntimeError('launch failed')
        with patch('aix_zero.checkpoint_publish.alive',return_value=False):
            with self.assertRaises(RuntimeError):resume_paused(self.state,self.repo,runner=runner)
        self.assertTrue((self.state/PAUSE).exists())
        result=resume_paused(self.state,self.repo,clear_only=True)
        self.assertFalse(result['resumed']);self.assertFalse((self.state/PAUSE).exists())

    def test_cli_does_not_resume_after_publish_failure(self):
        from aix_zero.aix_ctl import main
        with patch('aix_zero.checkpoint_publish.create_snapshot',return_value={}), \
             patch('aix_zero.checkpoint_publish.publish_release',side_effect=RuntimeError('upload failed')), \
             patch('aix_zero.checkpoint_publish.resume_paused') as resume:
            with self.assertRaises(RuntimeError):
                main(['--state-dir',str(self.state),'checkpoint','--publish','--resume'])
            resume.assert_not_called()
        atomic_json(self.state/PAUSE,dict(reason='test'))
        with self.assertRaisesRegex(RuntimeError,'pause'):
            main(['--state-dir',str(self.state),'start-daemon'])

    def test_real_physical_model_cartridges_and_tokenizer_roundtrip(self):
        self.model_patch.stop()
        import torch
        from aix_zero.benchmark_baseline import load_v11
        from aix_zero.candidate_checkpoint import encode_candidate_checkpoint, load_candidate_checkpoint
        from aix_zero.physical_lora import attach_physical_lora_to_model
        from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
        baseline=Path(__file__).resolve().parents[1]/'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        shutil.copyfile(baseline,self.baseline)
        model,tokenizer=load_v11(baseline);model.eval()
        attach_physical_lora_to_model(model,rank=4)
        (self.state/'aix_zero_v12_evolving.pt').write_bytes(encode_candidate_checkpoint(model,tokenizer))
        bank=SparseExpertCartridgeBank(self.state/'meta_ai_cartridges')
        bank.export_all_domain_cartridges(model,domain_catalog={'A':{},'B':{}},domain_accuracy={},domain_expert_map={'A':0,'B':1})
        snapshot=create_snapshot(self.state,self.repo,self.out)
        manifest=verify_bundle(snapshot['bundle'])
        self.assertEqual(manifest['model']['checkpoint_format'],'physical_lora_v2')
        self.assertGreater(manifest['model']['physical_accounting']['physical_lora_parameters'],0)
        restored,_,_=load_candidate_checkpoint((Path(snapshot['bundle'])/'state/aix_zero_v12_evolving.pt').read_bytes())
        ids=torch.tensor([[1,2,3]])
        restored.eval()
        with torch.no_grad():self.assertTrue(torch.equal(model(ids)['logits'],restored(ids)['logits']))
