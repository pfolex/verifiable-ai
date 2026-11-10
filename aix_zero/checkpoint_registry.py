"""Side-effect-free checkpoint loading and checkpoint provenance inventory (`BASE-05`, `BASE-06`).

Guarantees:
1. Importing this module or calling `load_checkpoint_strict()` creates no directories and
   never writes to any file or mutates embeddings with random weights.
2. Every committed `.safetensors` artifact in the repository is indexed in
   `research/checkpoint-inventory.json` with an immutable SHA-256 digest, physical parameter
   count, explicit provenance status, and visible provenance gaps.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator
from safetensors.torch import load_file, save_file
import torch

from artifact_io import canonical_json, write_json
from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.sparse_cartridge_bank import _atomic_write
from aix_zero.tokenizer import AIXZeroTokenizer, build_v11_tokenizer

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_SCHEMA = ROOT / "schemas/checkpoint-inventory-v1.schema.json"
EXPERIMENT_CONFIG_SCHEMA = ROOT / "schemas/experiment-config-v1.schema.json"
DEFAULT_INVENTORY_PATH = ROOT / "research/checkpoint-inventory.json"

CLI_COMMAND_CLASSIFICATION: Dict[str, Dict[str, str]] = {
    "aix_zero.benchmark_baseline": {
        "tier": "stable_release_interface",
        "purpose": "Offline v11 baseline evaluation and optional read-only expert cartridge export.",
        "network_or_paid_calls": "none",
    },
    "aix_zero.train_bayesian_pilot": {
        "tier": "experimental_research_command",
        "purpose": "Single-expert frozen-backbone Bayesian pilot training into an isolated output directory.",
        "network_or_paid_calls": "none",
    },
    "aix_zero.interactive_repl": {
        "tier": "experimental_research_command",
        "purpose": "Interactive REPL for inference, cartridge swapping, and in-RAM teaching (!save requires explicit output path).",
        "network_or_paid_calls": "optional_only_on_explicit_research_command",
    },
    "aix_zero.knowledge_is_power": {
        "tier": "experimental_research_command",
        "purpose": "Experimental multi-domain curriculum distillation and codex synthesis.",
        "network_or_paid_calls": "local_ollama_and_wikipedia_arxiv",
    },
}


def infer_config_from_state_dict(state: Dict[str, torch.Tensor]) -> AIXZeroConfig:
    """Infer exact `AIXZeroConfig` dimensions from a checkpoint state dict without guessing."""
    if "token_emb.weight" not in state or "lm_head.weight" not in state:
        raise ValueError("State dict is missing token_emb.weight or lm_head.weight")
    vocab_size, d_model = state["token_emb.weight"].shape
    if state["lm_head.weight"].shape != (vocab_size, d_model):
        raise ValueError("Untied or mismatched token_emb and lm_head shapes")

    use_moe = any(".moe." in k for k in state)
    use_shared_expert = any(".moe.shared_expert." in k for k in state)
    expert_indices = {
        int(k.split(".moe.experts.")[1].split(".")[0])
        for k in state
        if ".moe.experts." in k
    }
    n_experts = len(expert_indices) if expert_indices else 4
    use_memory_bank = any(".memory_bank." in k for k in state)
    use_prompt_pointer = any(".prompt_pointer." in k for k in state)
    use_key_ngram_conv = any(".prompt_pointer.key_conv." in k for k in state)

    n_recurrent_steps = 3
    if "latent_reasoning_core.step_embeddings.weight" in state:
        n_recurrent_steps = int(state["latent_reasoning_core.step_embeddings.weight"].shape[0])

    n_memory_slots = 8
    if "latent_reasoning_core.memory_bank.memory_keys" in state:
        n_memory_slots = int(state["latent_reasoning_core.memory_bank.memory_keys"].shape[0])

    return AIXZeroConfig(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_encoder_layers=1,
        n_recurrent_steps=n_recurrent_steps,
        n_decoder_layers=1,
        max_seq_len=112,
        use_moe=use_moe,
        use_shared_expert=use_shared_expert,
        n_experts=n_experts,
        moe_top_k=min(2, n_experts),
        use_memory_bank=use_memory_bank,
        n_memory_slots=n_memory_slots,
        use_prompt_pointer=use_prompt_pointer,
        pointer_temperature=0.35,
        use_key_ngram_conv=use_key_ngram_conv,
    )


def load_checkpoint_strict(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str = "cpu",
    tokenizer: Optional[AIXZeroTokenizer] = None,
    expected_tokenizer_metadata: Optional[Dict[str, Any]] = None,
    expected_sha256: Optional[str] = None,
    allow_append_only_superset: bool = True,
) -> Tuple[AIXZeroModel, AIXZeroTokenizer, Dict[str, Any]]:
    """Side-effect-free, strict checkpoint loader shared by REPL, benchmarks, and experiments.

    Never creates directories, never writes files, and never randomly expands embeddings.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    raw_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}"
        )

    state = load_file(str(path))
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError(f"Checkpoint {path.name} contains non-finite tensor values")

    config = infer_config_from_state_dict(state)
    model = AIXZeroModel(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    tok = tokenizer if tokenizer is not None else build_v11_tokenizer()
    tok.verify_checkpoint_compatibility(
        config.vocab_size,
        expected_tokenizer_metadata,
        allow_append_only_superset=allow_append_only_superset,
    )

    info = {
        "checkpoint_name": path.name,
        "sha256": actual_sha256,
        "serialized_bytes": len(raw_bytes),
        "physical_parameters": model.count_parameters()["total_parameters"],
        "vocab_size": config.vocab_size,
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_prefix_sha256": tok.prefix_sha256(config.vocab_size),
    }
    return model, tok, info


def inventory_digest(inventory: Dict[str, Any]) -> str:
    unsigned = {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def validate_checkpoint_inventory(
    inventory: Dict[str, Any],
    *,
    verify_files_on_disk: bool = True,
) -> None:
    schema = json.loads(INVENTORY_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(inventory)

    ids = [c["checkpoint_id"] for c in inventory["checkpoints"]]
    paths = [c["path"] for c in inventory["checkpoints"]]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Checkpoint inventory contains duplicate IDs or paths")

    for entry in inventory["checkpoints"]:
        if entry["provenance_status"] == "teacher_assisted_or_external_web_session":
            if not entry["known_provenance_gaps"]:
                raise ValueError(
                    f"Checkpoint {entry['checkpoint_id']!r} has teacher/web provenance status "
                    "and must list explicit known_provenance_gaps"
                )
        if verify_files_on_disk:
            file_path = ROOT / entry["path"]
            if not file_path.is_file():
                raise FileNotFoundError(f"Indexed checkpoint missing on disk: {entry['path']}")
            raw = file_path.read_bytes()
            if len(raw) != entry["serialized_bytes"]:
                raise ValueError(f"Byte size mismatch for {entry['path']}")
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {entry['path']}")

    expected_sha = inventory_digest(inventory)
    if inventory["inventory_sha256"] != expected_sha:
        raise ValueError("Checkpoint inventory SHA-256 mismatch")


def load_checkpoint_inventory(
    path: Path = DEFAULT_INVENTORY_PATH,
    *,
    verify_files_on_disk: bool = True,
) -> Dict[str, Any]:
    inventory = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_checkpoint_inventory(inventory, verify_files_on_disk=verify_files_on_disk)
    return inventory


def save_checkpoint_inventory(
    inventory: Dict[str, Any],
    path: Path = DEFAULT_INVENTORY_PATH,
) -> None:
    validate_checkpoint_inventory(inventory, verify_files_on_disk=True)
    write_json(inventory, path)


def _real_target_path(path: Path) -> Path:
    """Resolve symlinks on existing files or parent directories to detect aliased paths (`BASE-13`)."""
    p = Path(path)
    if p.exists() or p.is_symlink():
        return p.resolve()
    return p.parent.resolve() / p.name


def save_checkpoint_protected(
    model: AIXZeroModel,
    output_path: Path | str,
    *,
    source_checkpoint_path: Optional[Path | str] = None,
    allow_overwrite_source: bool = False,
    allow_overwrite_indexed: bool = False,
) -> Dict[str, Any]:
    """Save model weights atomically while protecting source and indexed baseline checkpoints (`BASE-13`).

    Raises PermissionError if `output_path` resolves (including via symlinks or `..`) to:
    - `source_checkpoint_path` (unless `allow_overwrite_source=True`), or
    - any checkpoint listed in `research/checkpoint-inventory.json` (unless `allow_overwrite_indexed=True`).
    """
    dest = Path(output_path)
    real_dest = _real_target_path(dest)

    if source_checkpoint_path is not None and not allow_overwrite_source:
        real_src = _real_target_path(Path(source_checkpoint_path))
        if real_dest == real_src:
            raise PermissionError(
                f"Refusing to overwrite source checkpoint {source_checkpoint_path} "
                "(pass allow_overwrite_source=True only for intentional in-place overrides)"
            )

    if not allow_overwrite_indexed and DEFAULT_INVENTORY_PATH.is_file():
        inv = json.loads(DEFAULT_INVENTORY_PATH.read_text(encoding="utf-8"))
        protected_real_paths = {
            _real_target_path(ROOT / entry["path"]) for entry in inv.get("checkpoints", [])
        }
        if real_dest in protected_real_paths:
            raise PermissionError(
                f"Refusing to overwrite indexed baseline checkpoint {dest} "
                "(pass allow_overwrite_indexed=True only for explicit baseline updates)"
            )

    state_cpu = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state_cpu.values()):
        raise ValueError("Refusing to save checkpoint with non-finite weights")

    from safetensors.torch import save
    payload = save(state_cpu)
    _atomic_write(dest, payload)
    return {
        "saved_path": str(dest),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "serialized_bytes": len(payload),
    }


def experiment_config_digest(config: Dict[str, Any]) -> str:
    unsigned = {k: v for k, v in config.items() if k != "config_sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate a machine-readable experiment configuration contract (`BASE-08`)."""
    canonical_json(config)
    schema = json.loads(EXPERIMENT_CONFIG_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)

    m_id = config["model_identity"]
    if m_id["trainable_scope"] == "single_routed_expert" and m_id.get("target_expert_slot") is None:
        raise ValueError("single_routed_expert scope requires an explicit integer target_expert_slot")
    if m_id["trainable_scope"] == "eval_only_frozen" and m_id["trainable_parameters"] != 0:
        raise ValueError("eval_only_frozen scope must have trainable_parameters == 0")

    expected = experiment_config_digest(config)
    if config["config_sha256"] != expected:
        raise ValueError(f"Experiment config_sha256 mismatch: expected {expected}, got {config['config_sha256']}")


def make_experiment_config(unsigned_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(unsigned_config)
    cfg["config_sha256"] = experiment_config_digest(cfg)
    validate_experiment_config(cfg)
    return cfg


def save_experiment_config(config: Dict[str, Any], path: Path | str) -> None:
    validate_experiment_config(config)
    write_json(config, path)


def load_experiment_config(path: Path | str) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_experiment_config(cfg)
    return cfg

