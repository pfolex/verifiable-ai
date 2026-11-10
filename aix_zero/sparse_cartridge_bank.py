"""Versioned storage and explicit installation of existing routed expert snapshots.

This module does not train specialists, route prompts, or make inference sparse.
Callers must serialize installation with model inference/training. The byte budget
covers cached CPU tensors only, not model slots or temporary deserialization memory.
"""
from __future__ import annotations

from collections import OrderedDict
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
