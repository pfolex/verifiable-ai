"""Verified cycle-boundary model bundles and opt-in GitHub release publication.

Release artifacts are distinct from Git commits and evaluated champions.
Importing this module performs no publication, restart or training.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from aix_zero.checkpoint_control import PAUSE, CYCLE_LOCK, PUBLISH_LOCK, atomic_json, file_lock, source_fingerprint


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def alive(pid):
    if type(pid) is not int or pid <= 0: return False
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def state_pid(state):
    path=state/'autopilot.pid'
    if not path.exists(): return None
    if path.is_symlink(): raise ValueError('PID file must not be a symlink')
    try: return int(path.read_text().strip())
    except ValueError: raise ValueError('Invalid daemon PID file') from None


def safe_relative(name):
    if not isinstance(name,str) or not name: raise ValueError('Unsafe bundle path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name: raise ValueError('Unsafe bundle path')
    return Path(*p.parts)


def regular(path, root):
    path=Path(path); root=Path(root).resolve()
    if not path.resolve().is_relative_to(root): raise ValueError('File escapes component root')
    for part in [path,*list(path.parents)[:len(path.parts)-len(root.parts)]]:
        if part.is_symlink(): raise ValueError('Snapshot components must not be symlinks')
    if not path.is_file(): raise FileNotFoundError('Missing snapshot component: '+str(path))
    return path


def _git(repo,*args):
    return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True).stdout


def model_info(path):
    from dataclasses import asdict
    from aix_zero.candidate_checkpoint import load_candidate_checkpoint
    from aix_zero.physical_lora import measure_physical_lora_Accounting
    model, tokenizer, fmt=load_candidate_checkpoint(Path(path).read_bytes())
    return dict(checkpoint_format=fmt,model_config=asdict(model.config),
                runtime_baseline_sha256=digest(Path(__file__).resolve().parent/'checkpoints/aix_zero_v11_kip.safetensors'),
                tokenizer=tokenizer.to_metadata(),physical_accounting=measure_physical_lora_Accounting(model))


def collect_components(state, repo, extra_state=(), bank_dir=None):
    components={}
    def add(path,name,root):
        regular(path,root); components[name]=path
    add(state/'aix_zero_v12_evolving.pt','state/aix_zero_v12_evolving.pt',state)
    for path in state.iterdir():
        if path.name.startswith('.') or path.name=='autopilot.pid': continue
        if path.suffix in ('.json','.pt','.safetensors') and path.is_file():
            add(path,'state/'+path.name,state)
    bank=Path(bank_dir) if bank_dir else state/'meta_ai_cartridges'
    if bank.exists():
        from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
        store=SparseExpertCartridgeBank(bank)
        manifest=store.get_manifest()
        if not store.manifest_path.is_file(): raise ValueError('Cartridge bank exists without manifest')
        add(store.manifest_path,'state/meta_ai_cartridges/cartridge_manifest.json',bank)
        for domain, entry in manifest['cartridges'].items():
            store._load(domain)  # Hash, byte length, finite tensor and schema checks.
            name=entry['cartridge_file']
            add(bank/name,'state/meta_ai_cartridges/'+name,bank)
    for name in extra_state:
        path=state/safe_relative(name)
        add(path,'state/'+safe_relative(name).as_posix(),state)
    baseline=repo/'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
    add(baseline,'runtime/aix_zero/checkpoints/aix_zero_v11_kip.safetensors',repo)
    return components


def verify_bundle(bundle):
    bundle=Path(bundle).resolve(); manifest=read_json(bundle/'manifest.json')
    if manifest.get('schema_version')!=1 or manifest.get('kind')!='latest_snapshot':
        raise ValueError('Unsupported checkpoint bundle')
    if any(p.is_symlink() for p in bundle.rglob('*')): raise ValueError('Bundle contains symlinks')
    expected=set(manifest['files'])|{'manifest.json'}
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual!=expected: raise ValueError('Missing or unexpected bundle files')
    for name,entry in manifest['files'].items():
        path=regular(bundle/safe_relative(name),bundle)
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            raise ValueError('Bundle integrity check failed: '+name)
    info=model_info(bundle/'state/aix_zero_v12_evolving.pt')
    if digest(bundle/'runtime/aix_zero/checkpoints/aix_zero_v11_kip.safetensors')!=info['runtime_baseline_sha256']:
        raise ValueError('Runtime baseline differs from the validating loader baseline')
    if info['model_config']!=manifest['model']['model_config'] or info['tokenizer']!=manifest['model']['tokenizer']:
        raise ValueError('Bundle model/tokenizer metadata differs')
    return manifest


def create_snapshot(state, repo, output, *, timeout=300, max_bytes=512*1024*1024, extra_state=()):
    state=Path(state).resolve(); repo=Path(repo).resolve(); output=Path(output).resolve()
    if not math.isfinite(timeout) or timeout<0 or type(max_bytes) is not int or max_bytes<=0:
        raise ValueError('Invalid snapshot budget')
    if output.is_relative_to(state): raise ValueError('Snapshot output must be outside the live state directory')
    output.mkdir(parents=True,exist_ok=True)
    with file_lock(state/PUBLISH_LOCK):
        status=read_json(state/'autopilot_status.json')
        pid=state_pid(state); was_running=alive(pid)
        if was_running and (status.get('checkpoint_protocol_version')!=1 or status.get('checkpoint_guard_pid')!=pid):
            raise RuntimeError('Running daemon lacks checkpoint boundary protocol; stop it manually and restart updated code first')
        lease=state/'candidate_bundle/registry/scheduler_worker.lock'
        if lease.exists() and alive(read_json(lease).get('pid')):
            raise RuntimeError('Stop the independent candidate scheduler before creating a daemon snapshot')
        if not (state/PAUSE).exists():
            resume_record=dict(was_running=was_running,pid=pid,
                launch=read_json(state/'.checkpoint.launch.json') if (state/'.checkpoint.launch.json').exists() else None)
            atomic_json(state/'.checkpoint.resume.json',resume_record)
        else:
            resume_record=read_json(state/'.checkpoint.resume.json')
        atomic_json(state/PAUSE,dict(schema_version=1,reason='CHECKPOINT_PUBLICATION',
                                   requested_at_utc=datetime.now(timezone.utc).isoformat()))
        with file_lock(state/CYCLE_LOCK,timeout):
            status=read_json(state/'autopilot_status.json')
            if status.get('checkpoint_cycle_phase') in ('active','failed','interrupted'):
                raise RuntimeError('Last cycle is incomplete/failed; inspect and recover before publishing')
            cycle=status.get('cycles_completed')
            if type(cycle) is not int or cycle<0: raise ValueError('Invalid completed cycle count')
            source_revision=_git(repo,'rev-parse','HEAD').decode().strip()
            if status.get('checkpoint_protocol_version')==1 and status.get('checkpoint_runtime_source_sha256')!=source_fingerprint(repo):
                raise RuntimeError('Runtime source changed since daemon start; restart updated code before publishing')
            changed=_git(repo,'diff','--name-only','HEAD').decode().splitlines()
            state_rel=state.relative_to(repo).as_posix()+'/' if state.is_relative_to(repo) else None
            if any(not state_rel or not name.startswith(state_rel) for name in changed):
                raise RuntimeError('Tracked source/config changes must be committed before a reproducible snapshot')
            tracked=_git(repo,'ls-files','-z').decode().split('\0')
            sources=[name for name in tracked if name and not name.startswith('aix_zero/autopilot_state/')
                     and (name.endswith(('.py','.toml','.yaml','.yml','.txt')) or name.startswith('schemas/') or name in ('LICENSE','README.md'))]
            if not sources: raise ValueError('No tracked source files available for the bundle')
            untracked=_git(repo,'ls-files','--others','--exclude-standard').decode().splitlines()
            if any((name.startswith('aix_zero/') and not name.startswith('aix_zero/autopilot_state/') or '/' not in name)
                   and name.endswith(('.py','.toml','.yaml','.yml')) for name in untracked):
                raise RuntimeError('Untracked runtime source files must be committed before snapshot')
            source_bytes=_git(repo,'archive','--format=tar',source_revision,'--',*sources)
            components=collect_components(state,repo,extra_state,status.get('checkpoint_cartridge_bank_dir'))
            if sum(p.stat().st_size for p in components.values())+len(source_bytes)>max_bytes:
                raise ValueError('Snapshot exceeds configured byte budget')
            tag=f'aix-cycle-{cycle}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-{uuid.uuid4().hex[:8]}'
            target=output/tag
            with tempfile.TemporaryDirectory(dir=output,prefix='.staging-') as temporary:
                bundle=Path(temporary)/'bundle';bundle.mkdir()
                files={}
                for name,path in components.items():
                    dest=bundle/safe_relative(name);dest.parent.mkdir(parents=True,exist_ok=True)
                    before=digest(path);shutil.copyfile(path,dest)
                    if digest(dest)!=before or digest(path)!=before: raise RuntimeError('Live component changed during snapshot')
                    files[name]=dict(sha256=before,bytes=dest.stat().st_size)
                (bundle/'source.tar').write_bytes(source_bytes)
                files['source.tar']=dict(sha256=digest(bundle/'source.tar'),bytes=len(source_bytes))
                manifest=dict(schema_version=1,kind='latest_snapshot',independently_evaluated_best=False,
                    tag=tag,cycle=cycle,source_revision=source_revision,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),files=files,
                    model=model_info(bundle/'state/aix_zero_v12_evolving.pt'),
                    daemon_launch=resume_record.get('launch'),
                    boundary='cooperative' if status.get('checkpoint_protocol_version')==1 else 'legacy_stopped_boundary_unconfirmed',
                    limitations=['Legacy daemon runtime snapshot; independent candidate scheduler/experiment registries are not included.',
                                 'Certification flags and lower validation loss are not independent capability confirmation.'])
                atomic_json(bundle/'manifest.json',manifest);verify_bundle(bundle)
                archive=Path(temporary)/(tag+'.zip')
                with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                    for path in sorted(bundle.rglob('*')):
                        if path.is_file(): z.write(path,path.relative_to(bundle).as_posix())
                if archive.stat().st_size>max_bytes: raise ValueError('Archive exceeds byte budget')
                with zipfile.ZipFile(archive) as z:
                    for name,entry in files.items():
                        if hashlib.sha256(z.read(name)).hexdigest()!=entry['sha256']: raise ValueError('Archive verification failed')
                    if hashlib.sha256(z.read('manifest.json')).hexdigest()!=digest(bundle/'manifest.json'):
                        raise ValueError('Archive manifest differs')
                target.mkdir()
                shutil.move(str(bundle),target/'bundle');shutil.move(str(archive),target/(tag+'.zip'))
            result=dict(tag=tag,cycle=cycle,source_revision=source_revision,bundle=str(target/'bundle'),
                archive=str(target/(tag+'.zip')),archive_sha256=digest(target/(tag+'.zip')),
                manifest_sha256=digest(target/'bundle/manifest.json'))
            atomic_json(target/'checksums.json',{k:result[k] for k in ('tag','cycle','source_revision','archive_sha256','manifest_sha256')})
            atomic_json(output/'latest.json',result)
            status.update(daemon_state='PAUSED_FOR_CHECKPOINT',checkpoint_cycle_phase='paused')
            atomic_json(state/'autopilot_status.json',status)
            return result


def publish_release(snapshot, repository, *, gh='gh', runner=subprocess.run):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository): raise ValueError('Expected owner/repository')
    archive=Path(snapshot['archive']); parent=archive.parent
    verify_bundle(snapshot['bundle'])
    if digest(archive)!=snapshot['archive_sha256'] or digest(Path(snapshot['bundle'])/'manifest.json')!=snapshot['manifest_sha256']:
        raise ValueError('Snapshot changed before publication')
    tag=snapshot['tag']; regular(archive,parent)
    checksums=read_json(parent/'checksums.json')
    if any(checksums.get(k)!=snapshot[k] for k in ('tag','cycle','source_revision','archive_sha256','manifest_sha256')):
        raise ValueError('Publication checksums differ from verified snapshot')
    notes=parent/'release-notes.md'
    notes.write_text(f"AIX cycle {snapshot['cycle']} runtime snapshot.\n\nSource: {snapshot['source_revision']}\n\nArchive SHA256: {snapshot['archive_sha256']}\n\nLatest snapshot, not an independently evaluated best model.\n")
    def run(*args):
        return runner([gh,*args],check=True,capture_output=True,text=True,timeout=300)
    run('release','create',tag,str(archive),str(parent/'checksums.json'),'--repo',repository,
        '--target',snapshot['source_revision'],'--title',f'AIX cycle {snapshot["cycle"]} checkpoint',
        '--notes-file',str(notes),'--draft','--latest=false')
    with tempfile.TemporaryDirectory(prefix='aix-release-verify-') as temporary:
        run('release','download',tag,'--repo',repository,'--pattern',archive.name,
            '--pattern','checksums.json','--dir',temporary)
        downloaded=Path(temporary)
        if digest(downloaded/archive.name)!=snapshot['archive_sha256'] or digest(downloaded/'checksums.json')!=digest(parent/'checksums.json'):
            raise ValueError('Downloaded GitHub assets differ; release remains draft and AIX remains paused')
    run('release','edit',tag,'--repo',repository,'--draft=false','--latest=false')
    receipt=dict(repository=repository,tag=tag,url=f'https://github.com/{repository}/releases/tag/{tag}',
                 archive_sha256=snapshot['archive_sha256'],verified_download=True)
    atomic_json(parent/'publication.json',receipt)
    atomic_json(parent.parent/'latest_published.json',receipt)
    return receipt


def resume_paused(state, repo, *, timeout=30, clear_only=False, runner=subprocess.run):
    state=Path(state).resolve();repo=Path(repo).resolve()
    with file_lock(state/PUBLISH_LOCK):
        if not (state/PAUSE).exists(): raise RuntimeError('No checkpoint pause to resume')
        record=read_json(state/'.checkpoint.resume.json')
        if record['was_running'] and not clear_only:
            end=time.monotonic()+timeout
            while alive(record['pid']):
                if time.monotonic()>=end: raise TimeoutError('Old daemon has not exited; pause retained')
                time.sleep(.1)
            launch=record.get('launch')
            if not isinstance(launch,dict): raise ValueError('No recorded daemon configuration; use --clear-only and restart explicitly')
            required={'max_cycles','initial_cycles_completed','interval_seconds','threads','sync_github','auto_push'}
            if set(launch)!=required: raise ValueError('Invalid recorded daemon configuration')
            for key in ('max_cycles','initial_cycles_completed','threads'):
                if type(launch[key]) is not int or launch[key]<0: raise ValueError('Invalid resume settings')
            if not 1<=launch['threads']<=256 or not 1<=launch['max_cycles']<=100000: raise ValueError('Resume settings exceed limits')
            if type(launch['interval_seconds']) not in (int,float) or not math.isfinite(launch['interval_seconds']) or launch['interval_seconds']<0:
                raise ValueError('Invalid resume interval')
            if any(type(launch[k]) is not bool for k in ('sync_github','auto_push')): raise ValueError('Invalid resume flags')
            completed=read_json(state/'autopilot_status.json')['cycles_completed']
            remaining=max(0,launch['max_cycles']-(completed-launch['initial_cycles_completed']))
        else: remaining=0
        with file_lock(state/CYCLE_LOCK,timeout):
            (state/PAUSE).unlink()
        try:
            if remaining:
                cmd=[sys.executable,'-m','aix_zero.aix_ctl','--state-dir',str(state),'start-daemon',
                     '--max-cycles',str(remaining),'--interval',str(launch['interval_seconds']),'--threads',str(launch['threads'])]
                if launch['sync_github']:cmd.append('--sync-github')
                if launch['auto_push']:cmd.append('--auto-push')
                runner(cmd,cwd=repo,check=True,capture_output=True,text=True,timeout=30)
        except BaseException:
            atomic_json(state/PAUSE,dict(reason='RESUME_FAILED'));raise
        if not remaining:
            status=read_json(state/'autopilot_status.json');status.update(daemon_state='STOPPED',checkpoint_cycle_phase='idle')
            atomic_json(state/'autopilot_status.json',status)
        return dict(resumed=bool(remaining),remaining_cycles=remaining,pause_cleared=True)
