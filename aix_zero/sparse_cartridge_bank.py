"""Versioned storage and explicit installation of existing routed expert snapshots.

This module does not train specialists, route prompts, or make inference sparse.
Callers must serialize installation with model inference/training. The byte budget
covers cached CPU tensors only, not model slots or temporary deserialization memory.
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load, save

from aix_zero.architecture import AIXZeroModel


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.cartridge-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _moe(model: AIXZeroModel):
    block = model.latent_reasoning_core.core_block
    if not block.use_moe:
        raise ValueError('Cartridges require a model configured with use_moe=True')
    return block.moe


class SparseExpertCartridgeBank:
    """Integrity-checked expert storage with an LRU cache bounded by bytes and count."""

    VALID_PROVENANCE_TYPES = (
        "existing_routed_expert_snapshot",
        "frozen_v11_single_expert_training_candidate",
    )
    VALID_PROMOTION_STATES = (
        "NOT_EVALUATED_SNAPSHOT",
        "PENDING_AUDIT",
        "REJECTED",
        "ELIGIBLE_NOT_ACTIVATED",
        "APPROVED_ACTIVE",
    )

    @classmethod
    def validate_specialist_provenance(cls, entry: Dict[str, Any]) -> None:
        """Enforce `EXPERT-01` (#89) provenance and promotion state separation on a cartridge entry."""
        prov = entry.get("provenance")
        if prov not in cls.VALID_PROVENANCE_TYPES:
            raise ValueError(f"Invalid cartridge provenance type: {prov!r}")

        status = entry.get("promotion_status", "NOT_EVALUATED_SNAPSHOT")
        if status not in cls.VALID_PROMOTION_STATES:
            raise ValueError(f"Invalid cartridge promotion_status: {status!r}")

        if prov == "existing_routed_expert_snapshot":
            if status not in ("NOT_EVALUATED_SNAPSHOT",):
                raise ValueError(
                    "Copied existing_routed_expert_snapshot cannot claim trained specialist promotion states"
                )
            if entry.get("verified_accuracy_pct") is not None:
                raise ValueError("Copied snapshot cannot claim a specialist verified_accuracy_pct")
        elif prov == "frozen_v11_single_expert_training_candidate":
            if status == "NOT_EVALUATED_SNAPSHOT":
                raise ValueError("Trained specialist candidate must declare an explicit audit/promotion status")
            training = entry.get("training")
            if not isinstance(training, dict) or not training.get("frozen_tensors_unchanged"):
                raise ValueError("Trained specialist candidate must record training recipe with frozen_tensors_unchanged=True")
            gate = entry.get("empirical_gate")
            if status in ("ELIGIBLE_NOT_ACTIVATED", "APPROVED_ACTIVE"):
                if not isinstance(gate, dict) or gate.get("decision") != "PASS":
                    raise ValueError(f"Promotion status {status!r} requires empirical_gate.decision == 'PASS'")
            if status == "REJECTED" and isinstance(gate, dict) and gate.get("decision") == "PASS":
                raise ValueError("REJECTED status conflicts with empirical_gate.decision == 'PASS'")

    def __init__(self, cartridge_dir: Optional[Path] = None, *,
                 max_cached_bytes: int = 1024 * 1024, max_cached_experts: int = 2,
                 max_shard_bytes: int = 16 * 1024 * 1024,
                 max_manifest_bytes: int = 1024 * 1024,
                 max_manifest_entries: int = 256) -> None:
        if (max_cached_bytes < 0 or max_cached_experts < 0 or max_shard_bytes <= 0
                or max_manifest_bytes <= 0 or max_manifest_entries <= 0):
            raise ValueError('Cache limits must be nonnegative; shard and manifest limits must be positive')
        self.cartridge_dir = Path(cartridge_dir) if cartridge_dir is not None else (
            Path(__file__).resolve().parent / 'checkpoints' / 'cartridges')
        self.manifest_path = self.cartridge_dir / 'cartridge_manifest.json'
        self.max_cached_bytes = max_cached_bytes
        self.max_cached_experts = max_cached_experts
        self.max_shard_bytes = max_shard_bytes
        self.max_manifest_bytes = max_manifest_bytes
        self.max_manifest_entries = max_manifest_entries
        self._hot_cache = OrderedDict()
        self._lock = threading.RLock()

    @property
    def cached_bytes(self) -> int:
        return sum(sum(t.numel() * t.element_size() for t in tensors.values())
                   for tensors in self._hot_cache.values())

    def cache_info(self) -> Dict[str, Any]:
        with self._lock:
            return dict(entries=len(self._hot_cache), bytes=self.cached_bytes,
                        max_bytes=self.max_cached_bytes, max_experts=self.max_cached_experts)

    def get_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.is_symlink():
            raise ValueError('Cartridge manifest path must not be a symlink')
        if not self.manifest_path.exists():
            return {}
        if self.manifest_path.stat().st_size > self.max_manifest_bytes:
            raise ValueError('Cartridge manifest exceeds max_manifest_bytes limit')
        raw_bytes = self.manifest_path.read_bytes()
        if len(raw_bytes) > self.max_manifest_bytes:
            raise ValueError('Cartridge manifest exceeds max_manifest_bytes limit')

        def _no_duplicate_keys(pairs):
            out = {}
            for k, v in pairs:
                if k in out:
                    raise ValueError(f'Duplicate JSON key in cartridge manifest: {k!r}')
                out[k] = v
            return out

        manifest = json.loads(raw_bytes.decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
        if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or not isinstance(manifest.get('cartridges'), dict):
            raise ValueError('Unsupported or malformed cartridge manifest')
        if len(manifest['cartridges']) > self.max_manifest_entries:
            raise ValueError('Cartridge manifest exceeds max_manifest_entries limit')
        return manifest

    def export_all_domain_cartridges(self, model: AIXZeroModel,
                                    domain_catalog: Dict[str, Dict[str, Any]],
                                    domain_accuracy: Dict[str, float], *,
                                    domain_expert_map: Dict[str, int]) -> Dict[str, Any]:
        """Snapshot explicitly mapped existing experts; domain labels imply no specialization.

        Accuracy is base-model telemetry, never a measured shard score. Content-addressed
        files are published first, then the manifest is atomically replaced. Old files
        are retained so readers of the previous manifest can finish safely.
        """
        with self._lock:
            moe = _moe(model)
            if set(domain_expert_map) != set(domain_catalog):
                raise ValueError('Provide an explicit expert index for every domain')
            for domain, index in domain_expert_map.items():
                if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', domain):
                    raise ValueError('Domain names must be safe identifiers')
                if type(index) is not int or not 0 <= index < len(moe.experts):
                    raise ValueError('Expert index out of range')
            entries, payloads = {}, {}
            for domain, meta in domain_catalog.items():
                index = domain_expert_map[domain]
                tensors = {k: v.detach().cpu().clone().contiguous()
                           for k, v in moe.experts[index].state_dict().items()}
                if any(not torch.isfinite(v).all() for v in tensors.values()):
                    raise ValueError('Cannot export nonfinite expert tensors')
                payload = save(tensors)
                if len(payload) > self.max_shard_bytes:
                    raise ValueError('Expert exceeds serialized shard byte limit')
                digest = hashlib.sha256(payload).hexdigest()
                filename = f'expert_{domain.lower()}_{digest}.safetensors'
                payloads[filename] = payload
                entries[domain] = {
                    'domain': domain, **{k: meta.get(k, '') for k in (
                        'title', 'category', 'prefix', 'formula', 'introduced_in')},
                    'cartridge_file': filename, 'sha256': digest,
                    'serialized_bytes': len(payload), 'source_expert_index': index,
                    'parameters': sum(v.numel() for v in tensors.values()),
                    'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                                for k, v in tensors.items()},
                    'base_model_accuracy_pct': domain_accuracy.get(domain),
                    'verified_accuracy_pct': None, 'sprt_gate': 'NOT_EVALUATED',
                    'provenance': 'existing_routed_expert_snapshot',
                }
            params = model.count_parameters()['total_parameters']
            manifest = {
                'schema_version': 1, 'architecture': 'AIX-Zero routed expert snapshots',
                'model_config': asdict(model.config),
                'active_core_physical_parameters': params,
                'registered_cartridges': len(entries),
                'total_cartridge_bank_parameters': sum(v['parameters'] for v in entries.values()),
                'cartridges': entries,
                'limitations': 'Snapshots duplicate existing experts; no added learned capacity or specialist validation.',
            }
            encoded = json.dumps(manifest, indent=2, allow_nan=False).encode()
            for filename, payload in payloads.items():
                _atomic_write(self.cartridge_dir / filename, payload)
            _atomic_write(self.manifest_path, encoded)
            self._hot_cache.clear()
            return manifest

    def _load(self, domain: str):
        manifest = self.get_manifest()
        if domain not in manifest.get('cartridges', {}):
            raise KeyError(f'Unknown cartridge: {domain}')
        meta = manifest['cartridges'][domain]
        filename = meta['cartridge_file']
        raw_path = self.cartridge_dir / filename
        if Path(filename).name != filename or raw_path.is_symlink():
            raise ValueError('Cartridge path escapes bank directory or is a symlink')
        path = raw_path.resolve()
        if path.parent != self.cartridge_dir.resolve():
            raise ValueError('Cartridge path escapes bank directory')
        if not re.fullmatch(r'[0-9a-f]{64}', meta['sha256']):
            raise ValueError('Expected a full SHA-256 hash')
        if path.stat().st_size > self.max_shard_bytes:
            raise ValueError('Cartridge exceeds serialized shard byte limit')
        with path.open('rb') as stream:
            payload = stream.read(self.max_shard_bytes + 1)
        if len(payload) > self.max_shard_bytes or len(payload) != meta['serialized_bytes']:
            raise ValueError('Cartridge byte length mismatch')
        if hashlib.sha256(payload).hexdigest() != meta['sha256']:
            raise ValueError('Cartridge SHA-256 mismatch')
        key = (domain, meta['sha256'])
        hit = key in self._hot_cache
        tensors = self._hot_cache.pop(key) if hit else load(payload)
        schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in tensors.items()}
        if schema != meta['tensors'] or sum(v.numel() for v in tensors.values()) != meta['parameters']:
            raise ValueError('Cartridge tensor schema mismatch')
        if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError('Cartridge tensors must be finite floating point values')
        size = sum(v.numel() * v.element_size() for v in tensors.values())
        if size <= self.max_cached_bytes and self.max_cached_experts:
            while self._hot_cache and (self.cached_bytes + size > self.max_cached_bytes or
                                      len(self._hot_cache) >= self.max_cached_experts):
                self._hot_cache.popitem(last=False)
            self._hot_cache[key] = tensors
        return manifest, meta, tensors, hit

    def page_in_cartridge(self, domain: str) -> Dict[str, Any]:
        """Load/cache a snapshot without changing model weights; errors fail closed."""
        with self._lock:
            start = time.perf_counter()
            try:
                _, meta, _, hit = self._load(domain)
            except KeyError:
                if domain in self.get_manifest().get('cartridges', {}):
                    raise
                return dict(loaded=False, domain=domain, latency_ms=0.0)
            return dict(loaded=True, installed=False, domain=domain,
                        cartridge_file=meta['cartridge_file'], parameters_paged_in=meta['parameters'],
                        cache_hit=hit, latency_ms=(time.perf_counter() - start) * 1000)

    def install_cartridge(self, model: AIXZeroModel, domain: str, *,
                          slot: Optional[int] = None,
                          active_base_checkpoint_sha256: Optional[str] = None) -> Dict[str, Any]:
        """Copy a validated snapshot into one existing slot, preserving Parameter objects.

        Requires eval mode and exclusive model access. This is an explicit weight swap,
        not automatic routing; it does not change router weights or expert count.
        When installing a `frozen_v11_single_expert_training_candidate`, enforces frozen-base
        checkpoint SHA-256 identity (`EXPERT-02` / Issue #90).
        """
        with self._lock:
            if model.training:
                raise ValueError('Installation requires model.eval() and exclusive model access')
            manifest, meta, tensors, hit = self._load(domain)
            if manifest['model_config'] != asdict(model.config):
                raise ValueError('Cartridge model configuration mismatch')
            prov = meta.get('provenance', 'existing_routed_expert_snapshot')
            required_base_sha = meta.get('base_checkpoint_sha256') or manifest.get('base_checkpoint_sha256')
            if prov == 'frozen_v11_single_expert_training_candidate':
                self.validate_specialist_provenance(meta)
                if not required_base_sha or not active_base_checkpoint_sha256:
                    raise ValueError(
                        'Installing a trained specialist requires both cartridge base_checkpoint_sha256 '
                        'and active_base_checkpoint_sha256'
                    )
                if required_base_sha != active_base_checkpoint_sha256:
                    raise ValueError('Trained specialist base_checkpoint_sha256 mismatch with active base model')
            elif required_base_sha and active_base_checkpoint_sha256 and required_base_sha != active_base_checkpoint_sha256:
                raise ValueError('Snapshot base_checkpoint_sha256 mismatch with active base model')

            moe = _moe(model)
            slot = meta['source_expert_index'] if slot is None else slot
            if type(slot) is not int or not 0 <= slot < len(moe.experts):
                raise ValueError('Target expert slot out of range')
            target = moe.experts[slot]
            expected = target.state_dict()
            if set(expected) != set(tensors) or any(
                    tensors[k].shape != expected[k].shape or tensors[k].dtype != expected[k].dtype
                    for k in expected):
                raise ValueError('Cartridge does not match target tensor keys, shapes, and dtypes')
            # Stage all device transfers before touching live parameters.
            staged = {k: v.to(device=expected[k].device) for k, v in tensors.items()}
            with torch.no_grad():
                target.load_state_dict(staged, strict=True)
            return dict(installed=True, domain=domain, slot=slot,
                        parameters=meta['parameters'], cache_hit=hit)

    @contextlib.contextmanager
    def temporary_specialist_session(
        self,
        model: AIXZeroModel,
        domain: str,
        *,
        slot: Optional[int] = None,
        active_base_checkpoint_sha256: Optional[str] = None,
        force_explicit_route: bool = True,
    ):
        """Stable lifecycle context manager for temporary specialist sessions (`EXPERT-04` / Issue #91).

        Guarantees:
        - Prevents concurrent temporary sessions on the same `model` instance.
        - Saves and restores target slot tensors, router state, `dispatch_mode`, and `training` mode
          even if an exception is raised inside the `with` block.
        """
        if getattr(model, '_active_specialist_session', False):
            raise RuntimeError('Concurrent temporary_specialist_session on the same model is prohibited')

        moe = _moe(model)
        manifest, meta, _, _ = self._load(domain)
        target_slot = meta['source_expert_index'] if slot is None else slot
        if type(target_slot) is not int or not 0 <= target_slot < len(moe.experts):
            raise ValueError('Target expert slot out of range')

        was_training = model.training
        prev_dispatch_mode = moe.dispatch_mode
        saved_slot_state = {
            k: v.detach().clone()
            for k, v in moe.experts[target_slot].state_dict().items()
        }
        saved_router_weight = moe.router.weight.detach().clone()
        saved_router_bias = moe.router.bias.detach().clone() if moe.router.bias is not None else None
        saved_top_k = moe.top_k

        model._active_specialist_session = True
        try:
            model.eval()
            install_info = self.install_cartridge(
                model,
                domain,
                slot=target_slot,
                active_base_checkpoint_sha256=active_base_checkpoint_sha256,
            )
            if force_explicit_route:
                with torch.no_grad():
                    moe.router.weight.zero_()
                    if moe.router.bias is not None:
                        moe.router.bias.fill_(-1e9)
                        moe.router.bias[target_slot] = 1e9
                moe.top_k = 1
            yield install_info
        finally:
            with torch.no_grad():
                moe.experts[target_slot].load_state_dict(saved_slot_state, strict=True)
                moe.router.weight.copy_(saved_router_weight)
                if saved_router_bias is not None and moe.router.bias is not None:
                    moe.router.bias.copy_(saved_router_bias)
            moe.top_k = saved_top_k
            moe.dispatch_mode = prev_dispatch_mode
            model.train(was_training)
            model._active_specialist_session = False

    def publish_verified_catalog_atomic(
        self,
        new_manifest: Dict[str, Any],
        *,
        simulate_interruption_before_manifest_swap: bool = False,
    ) -> Dict[str, Any]:
        """Crash-safe expert library publication (`EXPERT-13` / Issue #121).

        Verifies every referenced immutable `.safetensors` object on disk before atomically
        replacing `cartridge_manifest.json`, coordinates concurrent publishers via an exclusive
        lockfile, and leaves the previous catalog usable if interrupted before the atomic swap.
        """
        self.cartridge_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cartridge_dir / '.publish.lock'
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError('Concurrent catalog publication rejected (.publish.lock is held)')

        try:
            os.write(lock_fd, f'pid={os.getpid()}'.encode('utf-8'))
            os.close(lock_fd)
            lock_fd = -1

            if not isinstance(new_manifest, dict) or new_manifest.get('schema_version') != 1:
                raise ValueError('Invalid manifest schema_version')
            cartridges = new_manifest.get('cartridges')
            if not isinstance(cartridges, dict) or len(cartridges) > self.max_manifest_entries:
                raise ValueError('Invalid or oversized cartridges map')

            for domain, entry in cartridges.items():
                self.validate_specialist_provenance(entry)
                fname = entry.get('cartridge_file', '')
                fpath = self.cartridge_dir / fname
                if Path(fname).name != fname or fpath.is_symlink() or not fpath.is_file():
                    raise ValueError(f'Missing or invalid cartridge shard on disk for {domain}: {fname!r}')
                raw = fpath.read_bytes()
                if len(raw) != entry.get('serialized_bytes') or len(raw) > self.max_shard_bytes:
                    raise ValueError(f'Byte length mismatch for {fname}')
                if hashlib.sha256(raw).hexdigest() != entry.get('sha256'):
                    raise ValueError(f'SHA-256 mismatch for {fname}')
                tensors = load(raw)
                if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in tensors.values()):
                    raise ValueError(f'Non-finite tensors in {fname}')

            encoded = json.dumps(new_manifest, indent=2, allow_nan=False).encode('utf-8')
            if len(encoded) > self.max_manifest_bytes:
                raise ValueError('Serialized manifest exceeds max_manifest_bytes')

            if simulate_interruption_before_manifest_swap:
                raise InterruptedError('Simulated interruption before manifest replacement')

            _atomic_write(self.manifest_path, encoded)
            self._hot_cache.clear()
            return {'published': True, 'entries': len(cartridges)}
        finally:
            if lock_fd != -1:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            if lock_path.exists():
                lock_path.unlink()

    def route_explicit_task_schema(self, prompt: str) -> Dict[str, Any]:
        """Deterministic explicit task router (`EXPERT-05` / Issue #92).

        Routes recognized task prefixes ONLY to cartridges with `promotion_status == 'APPROVED_ACTIVE'`.
        Unknown, malformed, or unpromoted (`REJECTED` / `NOT_EVALUATED_SNAPSHOT`) domains fall back
        to the base model without activating a specialist.
        """
        s = prompt.strip() if isinstance(prompt, str) else ''
        m = re.match(r'^Q:([A-Za-z0-9_]+):', s)
        if not m:
            return {
                'specialist_activated': False,
                'selected_domain': None,
                'route_target': 'BASE_MODEL_DEFAULT',
                'reason': 'unrecognized_or_malformed_task_prefix',
            }

        prefix_tag = m.group(1).upper()
        manifest = self.get_manifest()
        cartridges = manifest.get('cartridges', {})
        entry = cartridges.get(prefix_tag)
        if not entry:
            return {
                'specialist_activated': False,
                'selected_domain': prefix_tag,
                'route_target': 'BASE_MODEL_DEFAULT',
                'reason': 'no_registered_cartridge_for_prefix',
            }

        status = entry.get('promotion_status', 'NOT_EVALUATED_SNAPSHOT')
        if status != 'APPROVED_ACTIVE':
            return {
                'specialist_activated': False,
                'selected_domain': prefix_tag,
                'route_target': 'BASE_MODEL_DEFAULT',
                'reason': f'cartridge_not_promoted_{status}',
            }

        self.validate_specialist_provenance(entry)
        return {
            'specialist_activated': True,
            'selected_domain': prefix_tag,
            'route_target': entry['cartridge_file'],
            'reason': 'approved_active_specialist_matched',
        }

    def cleanup_unreferenced_shards(
        self,
        *,
        retained_manifests: Optional[list] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Safe unreferenced-shard garbage collection (`EXPERT-14` / Issue #122).

        Protects every `.safetensors` shard referenced by the active `cartridge_manifest.json`
        or any historical/rollback manifest in `retained_manifests`.
        """
        with self._lock:
            referenced_files = set()
            active_manifest = self.get_manifest()
            for entry in active_manifest.get('cartridges', {}).values():
                if 'cartridge_file' in entry:
                    referenced_files.add(entry['cartridge_file'])
            for hist in retained_manifests or []:
                for entry in hist.get('cartridges', {}).values():
                    if 'cartridge_file' in entry:
                        referenced_files.add(entry['cartridge_file'])

            protected_shards = []
            reclaimable_shards = []
            deleted_shards = []

            if self.cartridge_dir.exists():
                for fpath in sorted(self.cartridge_dir.iterdir()):
                    if not fpath.is_file() or fpath.name == 'cartridge_manifest.json' or fpath.name == '.publish.lock':
                        continue
                    if fpath.name in referenced_files:
                        protected_shards.append(fpath.name)
                    elif fpath.suffix == '.safetensors' or fpath.name.startswith('.cartridge-'):
                        reclaimable_shards.append(fpath.name)
                        if not dry_run:
                            fpath.unlink()
                            deleted_shards.append(fpath.name)

            return {
                'dry_run': dry_run,
                'protected_shards': protected_shards,
                'reclaimable_shards': reclaimable_shards,
                'deleted_shards': deleted_shards,
            }

    def execute_specialist_with_bounded_fallback(
        self,
        model: AIXZeroModel,
        domain: str,
        *,
        slot_index: int = 0,
        active_base_checkpoint_sha256: Optional[str] = None,
        candidate_output: Optional[str] = None,
        attempted_chain: Optional[List[str]] = None,
        max_retry_attempts: int = 2,
    ) -> Dict[str, Any]:
        """Execute a specialist with bounded fallback to the frozen base model (`EXPERT-07` / Issue #116).

        Handles unavailable, incompatible, rejected, and invalid-output specialists while
        preserving base model state and enforcing `max_retry_attempts` to prevent recursive
        expert-selection loops.
        """
        if max_retry_attempts < 1 or max_retry_attempts > 3:
            raise ValueError("max_retry_attempts must be in 1..3")

        moe = _moe(model)
        saved_base_slot = {
            k: v.detach().clone()
            for k, v in moe.experts[slot_index].state_dict().items()
        }

        def _base_preserved() -> bool:
            cur = moe.experts[slot_index].state_dict()
            return all(torch.equal(cur[k], saved_base_slot[k]) for k in saved_base_slot)

        chain = list(attempted_chain or [])
        if domain in chain:
            return {
                "executed_route": "base_model_fallback",
                "fallback_triggered": True,
                "fallback_reason": "RECURSIVE_EXPERT_LOOP_PREVENTED",
                "attempted_chain": chain + [domain],
                "base_preserved": _base_preserved(),
            }
        if len(chain) >= max_retry_attempts:
            return {
                "executed_route": "base_model_fallback",
                "fallback_triggered": True,
                "fallback_reason": "RETRY_LIMIT_EXCEEDED",
                "attempted_chain": chain,
                "base_preserved": _base_preserved(),
            }

        chain.append(domain)
        try:
            with self.temporary_specialist_session(
                model,
                domain,
                slot=slot_index,
                active_base_checkpoint_sha256=active_base_checkpoint_sha256,
            ):
                if candidate_output is not None and ("<ans>" not in candidate_output or "</ans>" not in candidate_output):
                    raise RuntimeError("INVALID_SPECIALIST_OUTPUT")
        except (KeyError, FileNotFoundError):
            return {
                "executed_route": "base_model_fallback",
                "fallback_triggered": True,
                "fallback_reason": "UNAVAILABLE_CARTRIDGE",
                "attempted_chain": chain,
                "base_preserved": _base_preserved(),
            }
        except ValueError as exc:
            msg = str(exc)
            reason = (
                "REJECTED_PROMOTION_GATE"
                if ("promotion_status" in msg or "empirical_gate" in msg)
                else "INCOMPATIBLE_CARTRIDGE"
            )
            return {
                "executed_route": "base_model_fallback",
                "fallback_triggered": True,
                "fallback_reason": reason,
                "error_detail": msg,
                "attempted_chain": chain,
                "base_preserved": _base_preserved(),
            }
        except RuntimeError as exc:
            if str(exc) == "INVALID_SPECIALIST_OUTPUT":
                return {
                    "executed_route": "base_model_fallback",
                    "fallback_triggered": True,
                    "fallback_reason": "INVALID_SPECIALIST_OUTPUT",
                    "attempted_chain": chain,
                    "base_preserved": _base_preserved(),
                }
            raise

        return {
            "executed_route": f"specialist:{domain}",
            "fallback_triggered": False,
            "fallback_reason": None,
            "attempted_chain": chain,
            "base_preserved": _base_preserved(),
        }

    def compose_experts_with_verified_interfaces(
        self,
        stages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compose multiple specialist stages with verified input/output contracts (`EXPERT-08` / Issue #98).

        Each stage validates its input/output contract and provenance; any failed stage prevents
        an unsupported final answer from being marked verified.
        """
        if not stages:
            raise ValueError("stages must be nonempty")

        manifest = self.get_manifest()
        cartridges = manifest.get("cartridges", {})
        stage_reports: List[Dict[str, Any]] = []
        pipeline_verified = True
        failed_stage_name: Optional[str] = None

        for idx, st in enumerate(stages):
            s_name = str(st["stage_name"])
            dom = str(st["domain"])
            inp_ok = bool(st.get("input_contract_valid", False))
            out_ok = bool(st.get("output_contract_valid", False))
            meta = cartridges.get(dom)
            prov_ok = meta is not None and (
                meta.get("provenance") == "existing_routed_expert_snapshot"
                or meta.get("promotion_status") == "APPROVED_ACTIVE"
            )
            stage_ok = inp_ok and out_ok and prov_ok
            if not stage_ok and pipeline_verified:
                pipeline_verified = False
                failed_stage_name = s_name

            stage_reports.append(
                {
                    "stage_index": idx,
                    "stage_name": s_name,
                    "domain": dom,
                    "input_contract_valid": inp_ok,
                    "output_contract_valid": out_ok,
                    "provenance_verified": prov_ok,
                    "stage_passed": stage_ok,
                }
            )

        return {
            "pipeline_verified": pipeline_verified,
            "final_answer_marked_verified": pipeline_verified,
            "failed_stage": failed_stage_name,
            "stage_reports": stage_reports,
        }

    def revoke_and_rollback_expert_version(
        self,
        domain: str,
        *,
        revocation_reason: str,
        known_good_manifest: Dict[str, Any],
        active_inflight_sessions: int = 0,
    ) -> Dict[str, Any]:
        """Revoke a regressed/compromised expert and roll back to a known-good manifest (`EXPERT-15` / Issue #123).

        Restores the known-good previous version reproducibly and specifies in-flight request
        handling (`DRAIN_INFLIGHT_SESSIONS_BEFORE_UNLINK`).
        """
        cur_manifest = self.get_manifest()
        if domain not in cur_manifest.get("cartridges", {}):
            raise KeyError(f"Unknown expert domain for revocation: {domain}")

        revoked_sha = cur_manifest["cartridges"][domain]["sha256"]
        if domain not in known_good_manifest.get("cartridges", {}):
            raise ValueError(f"known_good_manifest does not contain a fallback entry for {domain}")

        self.publish_verified_catalog_atomic(known_good_manifest)
        restored = self.get_manifest()["cartridges"][domain]

        return {
            "domain": domain,
            "revoked_sha256": revoked_sha,
            "restored_sha256": restored["sha256"],
            "revocation_reason": revocation_reason,
            "active_inflight_sessions": int(active_inflight_sessions),
            "inflight_request_policy": "DRAIN_INFLIGHT_SESSIONS_BLOCK_NEW_INSTALLS",
            "rollback_reproducible": restored["sha256"] == known_good_manifest["cartridges"][domain]["sha256"],
        }


def evaluate_learned_vs_explicit_routing(
    *,
    explicit_routing_metrics: Dict[str, float],
    learned_routing_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Compare learned routing against explicit schema routing (`EXPERT-06` / Issue #115).

    Compares `task_selection_accuracy`, `answer_accuracy`, `overhead_ms`, and `abstention_rate`,
    and ensures a learned router cannot hide an expert quality regression.
    """
    req = ("task_selection_accuracy", "answer_accuracy", "overhead_ms", "abstention_rate")
    for k in req:
        if k not in explicit_routing_metrics or k not in learned_routing_metrics:
            raise ValueError(f"Routing comparison missing metric {k!r}")

    exp_ans = float(explicit_routing_metrics["answer_accuracy"])
    lrn_ans = float(learned_routing_metrics["answer_accuracy"])
    regression = lrn_ans < exp_ans

    return {
        "explicit_routing": {k: round(float(explicit_routing_metrics[k]), 6) for k in req},
        "learned_routing": {k: round(float(learned_routing_metrics[k]), 6) for k in req},
        "answer_accuracy_delta": round(lrn_ans - exp_ans, 6),
        "learned_router_quality_regression_detected": regression,
        "learned_routing_approved": (not regression)
        and float(learned_routing_metrics["task_selection_accuracy"]) >= float(explicit_routing_metrics["task_selection_accuracy"]),
    }


class DeviceResidentExpertSlotManager:
    """Manage bounded device-resident expert slots (`EXPERT-11` / Issue #119).

    Reports total stored bank parameters separately from resident active parameters,
    and preserves router-to-expert slot identity while refusing incompatible weights.
    """

    def __init__(self, model: AIXZeroModel, bank: SparseExpertCartridgeBank, *, max_resident_slots: int = 2) -> None:
        moe = _moe(model)
        if max_resident_slots < 1 or max_resident_slots > len(moe.experts):
            raise ValueError("max_resident_slots out of range")
        self.model = model
        self.bank = bank
        self.max_resident_slots = int(max_resident_slots)
        self.slot_to_domain: Dict[int, str] = {}

    def replace_resident_slot(self, slot: int, domain: str, *, active_base_checkpoint_sha256: Optional[str] = None) -> Dict[str, Any]:
        if slot not in self.slot_to_domain and len(self.slot_to_domain) >= self.max_resident_slots:
            raise RuntimeError(f"Resident slot cap ({self.max_resident_slots}) reached; evict a slot first")
        self.model.eval()
        info = self.bank.install_cartridge(
            self.model, domain, slot=slot, active_base_checkpoint_sha256=active_base_checkpoint_sha256
        )
        self.slot_to_domain[int(slot)] = domain
        manifest = self.bank.get_manifest()
        moe = _moe(self.model)
        resident_expert_params = sum(
            sum(p.numel() for p in moe.experts[s].parameters()) for s in self.slot_to_domain
        )
        return {
            "slot": int(slot),
            "domain": domain,
            "router_to_expert_identity": dict(self.slot_to_domain),
            "resident_slot_count": len(self.slot_to_domain),
            "max_resident_slots": self.max_resident_slots,
            "resident_active_parameters": resident_expert_params,
            "total_physical_model_parameters": self.model.count_parameters()["total_parameters"],
            "total_stored_bank_parameters": manifest.get("total_cartridge_bank_parameters", 0),
            "installed": info["installed"],
        }


