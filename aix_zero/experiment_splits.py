"""Versioned split manifests and training-ingestion guards for autonomous experiments.

Identity functions are explicit: hashing alone cannot discover semantic duplicates.
These guards must be called by training, replay and retrieval ingestion paths.
"""
import hashlib
import json
from copy import deepcopy

SPLITS = ('train', 'development', 'regression', 'final')


def build_split_manifest(splits, identity, *, family_identity=None):
    if set(splits) != set(SPLITS):
        raise ValueError('All four split names are required')
    owners = {}; families = {}; records = {}
    for split in SPLITS:
        records[split] = []
        for row in splits[split]:
            key = identity(row)
            if not isinstance(key, str) or not key:
                raise ValueError('Nonempty semantic identity required')
            if key in owners:
                raise ValueError('Duplicate semantic identity across or within splits')
            owners[key] = split
            family = family_identity(row) if family_identity else None
            if family_identity and (not isinstance(family, str) or not family):
                raise ValueError('Nonempty family identity required')
            if family is not None:
                families.setdefault(family, set()).add(split)
            records[split].append(dict(identity=key, family=family, row=deepcopy(row)))
    if any('final' in owners and len(owners) > 1 for owners in families.values()):
        raise ValueError('Reserved final family overlaps another split')
    payload = dict(schema_version=1, family_isolated_final=family_identity is not None, splits=records)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(payload=payload, sha256=hashlib.sha256(encoded).hexdigest())


def verify_manifest(manifest):
    payload = manifest['payload']
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest['sha256']:
        raise ValueError('Split manifest digest mismatch')
    if payload['schema_version'] != 1 or set(payload['splits']) != set(SPLITS):
        raise ValueError('Unsupported split manifest')
    seen = set()
    for split in SPLITS:
        for record in payload['splits'][split]:
            key = record['identity']
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError('Invalid or duplicate semantic identity')
            seen.add(key)
    return deepcopy(payload)


def eligible_training_rows(manifest, rows, identity):
    """Fail closed for unregistered or held-out examples, including legacy memory."""
    payload = verify_manifest(manifest)
    allowed = {r['identity'] for r in payload['splits']['train']}
    result = []
    for row in rows:
        if identity(row) not in allowed:
            raise ValueError('Unregistered or evaluation example in training input')
        # Require exact registered content as well: matching IDs cannot smuggle new targets.
        if not any(r['identity'] == identity(row) and r['row'] == row for r in payload['splits']['train']):
            raise ValueError('Training content differs from registered manifest')
        result.append(deepcopy(row))
    return result
