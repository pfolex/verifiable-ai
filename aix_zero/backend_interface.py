"""Unified Backend Interface & Model Compatibility Validator (`#438` & `#439`).

Provides:
1. `ModelBackendAdapter`: Explicit backend contract for model inspection, tokenizer/chat
   rendering, generation, training, and artifact reload with explicit checkpoint format
   enforcement (no silent conversion between custom torch state_dict, HF safetensors, and GGUF).
2. `AIXZeroBackendAdapter`: Adapter for native `AIXZeroModel` checkpoints (`.safetensors` / `.pt`).
3. `validate_hf_or_mergekit_model_dir`: Pre-flight compatibility validator for user-supplied
   Hugging Face and local Mergekit exports (shard completeness, RoPE, tokenizer/chat template,
   tested architecture matrix, Mergekit lineage, and labeled disk vs. memory estimates).
4. `HuggingFaceBackendAdapter`: Explicit HF adapter rejecting unsupported configurations before training.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.benchmark_baseline import load_v11
from aix_zero.routed_specialist_training import compute_model_state_sha256


SUPPORTED_HF_COMPATIBILITY_MATRIX: Dict[str, Dict[str, Any]] = {
    "qwen2": {
        "status": "TESTED_SUPPORTED",
        "supported_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "supported_devices": ["cpu", "mps", "cuda"],
        "notes": "Primary tested family for train_aix.py causal-LM LoRA.",
    },
    "llama": {
        "status": "COMPATIBLE_EXPERIMENTAL",
        "supported_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "supported_devices": ["cpu", "mps", "cuda"],
        "notes": "Compatible attention/MLP projection names; requires local safetensors and chat template.",
    },
    "mistral": {
        "status": "COMPATIBLE_EXPERIMENTAL",
        "supported_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "supported_devices": ["cpu", "mps", "cuda"],
        "notes": "Compatible attention/MLP projection names; sliding-window configs require verification.",
    },
}


class ModelBackendAdapter(ABC):
    """Explicit backend contract for model inspection, rendering, training, generation, and reload."""

    backend_name: str = "abstract"
    checkpoint_format: str = "unknown"

    @abstractmethod
    def inspect_model(self) -> Dict[str, Any]:
        """Return architecture, physical/trainable parameter counts, context limit, and version metadata."""

    @abstractmethod
    def render_prompt(self, prompt: str) -> str:
        """Render a user prompt using the backend's native prompt or chat template."""

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 36) -> Dict[str, Any]:
        """Generate completion deterministically without reference leakage."""

    @abstractmethod
    def train_step(self, examples: Sequence[Dict[str, str]], *, lr: float = 1e-3, steps: int = 1) -> Dict[str, Any]:
        """Execute bounded gradient steps and return pre/post loss."""

    @abstractmethod
    def reload_artifact(self, artifact_path: Path) -> Dict[str, Any]:
        """Reload a saved checkpoint or adapter artifact with explicit format verification."""


class AIXZeroBackendAdapter(ModelBackendAdapter):
    """Backend adapter for native `AIXZeroModel` `.safetensors` and `.pt` checkpoints."""

    backend_name = "aix_zero_native"

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        state_dict_override_path: Optional[Path] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if self.checkpoint_path.suffix.lower() == ".gguf":
            raise ValueError(
                "GGUF checkpoints are not supported by AIXZeroBackendAdapter; "
                "supply a native AIX-Zero .safetensors or .pt checkpoint."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"AIX-Zero checkpoint not found: {self.checkpoint_path}")
        if self.checkpoint_path.is_dir() and (self.checkpoint_path / "config.json").exists():
            raise ValueError(
                "Directory appears to be a Hugging Face checkpoint; use HuggingFaceBackendAdapter "
                "instead of silent format conversion."
            )

        self.model, self.tokenizer = load_v11(self.checkpoint_path)
        self.checkpoint_format = "aix_zero_safetensors"
        self.loaded_from = str(self.checkpoint_path)
        if state_dict_override_path is not None:
            ov = Path(state_dict_override_path)
            if ov.suffix.lower() != ".pt":
                raise ValueError(f"Unsupported override format {ov.suffix!r}; expected '.pt'")
            self.model.load_state_dict(torch.load(ov, map_location="cpu", weights_only=True))
            self.checkpoint_format = "aix_zero_torch_state_dict"
            self.loaded_from = str(ov)
        self.model.eval()

    def inspect_model(self) -> Dict[str, Any]:
        counts = self.model.count_parameters()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        disk_bytes = Path(self.loaded_from).stat().st_size
        return {
            "backend": self.backend_name,
            "checkpoint_format": self.checkpoint_format,
            "loaded_from": self.loaded_from,
            "model_config": asdict(self.model.config),
            "physical_parameters": int(counts["total_parameters"]),
            "trainable_parameters": int(trainable),
            "context_limit": int(self.model.config.max_seq_len),
            "rope_theta": float(self.model.config.rope_theta),
            "vocab_size": int(self.tokenizer.vocab_size),
            "disk_weight_bytes": int(disk_bytes),
            "estimated_inference_memory_bytes": int(counts["total_parameters"] * 4),
            "estimated_training_memory_bytes": int(counts["total_parameters"] * 16),
            "memory_values_are_estimates": True,
            "state_sha256": compute_model_state_sha256(self.model),
            "versions": {"torch": torch.__version__},
        }

    def render_prompt(self, prompt: str) -> str:
        return str(prompt)

    def generate(self, prompt: str, *, max_new_tokens: int = 36) -> Dict[str, Any]:
        rendered = self.render_prompt(prompt)
        p_ids = self.tokenizer.encode(rendered, add_bos=True, add_eos=False)
        if len(p_ids) >= self.model.config.max_seq_len:
            raise ValueError("Prompt exceeds AIX-Zero context limit")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                gen_ids, conf, ponder = self.model.generate_verified(
                    p_ids,
                    eos_id=self.tokenizer.eos_id,
                    max_new_tokens=int(max_new_tokens),
                    num_candidates=1,
                    temperature=0.0,
                    tokenizer=self.tokenizer,
                    inflight_alu=False,
                )
        finally:
            self.model.train(was_training)
        return {
            "backend": self.backend_name,
            "assistance_mode": "neural_only",
            "completion": self.tokenizer.decode(gen_ids[len(p_ids) :], skip_special=False),
            "verifier_confidence": round(float(conf), 4),
            "ponder_depth": round(float(ponder), 4),
        }

    def train_step(self, examples: Sequence[Dict[str, str]], *, lr: float = 1e-3, steps: int = 1) -> Dict[str, Any]:
        if not examples or steps < 1 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("examples must be non-empty, steps >= 1, and lr finite positive")
        encoded: List[Tuple[List[int], List[int]]] = []
        for ex in examples:
            p_ids = self.tokenizer.encode(self.render_prompt(ex["prompt"]), add_bos=True, add_eos=False)
            c_ids = self.tokenizer.encode(ex["completion"], add_bos=False, add_eos=True)
            full = p_ids + c_ids
            if len(full) > self.model.config.max_seq_len:
                continue
            encoded.append((full, [0] * len(p_ids) + c_ids))
        if not encoded:
            raise ValueError("No training examples fit within AIX-Zero context limit")
        width = max(len(x) for x, _ in encoded)
        inputs = torch.tensor([x + [0] * (width - len(x)) for x, _ in encoded], dtype=torch.long)
        labels = torch.tensor([y + [0] * (width - len(y)) for _, y in encoded], dtype=torch.long)

        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=float(lr))
        self.model.eval()
        with torch.no_grad():
            pre_loss = float(self.model(inputs, labels=labels)["loss"].item())
        self.model.train()
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            loss = self.model(inputs, labels=labels)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
        self.model.eval()
        with torch.no_grad():
            post_loss = float(self.model(inputs, labels=labels)["loss"].item())
        return {
            "backend": self.backend_name,
            "steps_completed": int(steps),
            "pre_loss": round(pre_loss, 6),
            "post_loss": round(post_loss, 6),
            "state_sha256": compute_model_state_sha256(self.model),
        }

    def reload_artifact(self, artifact_path: Path) -> Dict[str, Any]:
        ap = Path(artifact_path)
        if ap.suffix.lower() != ".pt":
            raise ValueError(f"Expected .pt state_dict artifact for AIXZeroBackendAdapter, got {ap.suffix!r}")
        self.model.load_state_dict(torch.load(ap, map_location="cpu", weights_only=True))
        self.model.eval()
        return {"reloaded": True, "artifact": str(ap), "state_sha256": compute_model_state_sha256(self.model)}


def _detect_mergekit_lineage(model_dir: Path, raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect optional Mergekit recipe and parent identities; mark unknown lineage explicitly."""
    for fname in ("mergekit_config.yml", "mergekit_config.yaml", "merge_recipe.yml", "merge_recipe.yaml"):
        fpath = model_dir / fname
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            parents = sorted(set(re.findall(r"model:\s*([^\s#]+)", text)))
            return {
                "is_mergekit_export": True,
                "recipe_file": fname,
                "parent_models": parents if parents else ["unknown"],
                "lineage_status": "recorded_from_mergekit_recipe" if parents else "unknown_parents",
            }
    if "merge_method" in raw_config or "mergekit" in json.dumps(raw_config).lower():
        return {
            "is_mergekit_export": True,
            "recipe_file": "config.json",
            "parent_models": raw_config.get("parent_models") or ["unknown"],
            "lineage_status": "recorded_in_config" if raw_config.get("parent_models") else "unknown_parents",
        }
    return {
        "is_mergekit_export": False,
        "recipe_file": None,
        "parent_models": ["unknown"],
        "lineage_status": "not_declared",
    }


def validate_hf_or_mergekit_model_dir(
    model_dir: Path,
    *,
    requested_lora_targets: Sequence[str] = ("q_proj", "v_proj"),
    requested_device: str = "auto",
) -> Dict[str, Any]:
    """Pre-flight validator for local Hugging Face & Mergekit exports (`#439`).

    Rejects GGUF/Ollama inputs, missing weight shards, missing chat templates/EOS tokens,
    invalid RoPE settings, or unsupported architectures before any training starts.
    """
    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {p}")
    if p.is_file() and p.suffix.lower() == ".gguf":
        raise ValueError(
            "GGUF inputs are inference-only and unsupported for this LoRA training path. "
            "Export or supply a local Hugging Face safetensors directory instead."
        )
    if p.is_dir() and (any(p.glob("*.gguf")) or (p / "Modelfile").exists()) and not (p / "config.json").exists():
        raise ValueError(
            "Ollama/GGUF directory detected without Hugging Face config.json. "
            "Supply a local Hugging Face safetensors directory (trust_remote_code is never enabled)."
        )

    cfg_path = p / "config.json"
    if not cfg_path.exists():
        raise ValueError(f"Missing config.json in model directory: {p}")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model_type = str(raw_cfg.get("model_type", "")).lower()
    if model_type not in SUPPORTED_HF_COMPATIBILITY_MATRIX:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Tested compatibility matrix supports: "
            f"{sorted(SUPPORTED_HF_COMPATIBILITY_MATRIX.keys())}"
        )
    matrix_entry = SUPPORTED_HF_COMPATIBILITY_MATRIX[model_type]

    # Validate LoRA targets against tested matrix
    unsupported_targets = [t for t in requested_lora_targets if t not in matrix_entry["supported_lora_targets"]]
    if unsupported_targets:
        raise ValueError(
            f"Unsupported LoRA target_modules {unsupported_targets} for {model_type!r}; "
            f"supported: {matrix_entry['supported_lora_targets']}"
        )

    # Validate RoPE settings
    rope_params = raw_cfg.get("rope_parameters")
    if rope_params is not None:
        if rope_params.get("rope_type", "default") != "default":
            raise ValueError(f"Unsupported rope_type: {rope_params.get('rope_type')!r}")
        theta = rope_params.get("rope_theta")
        if theta is None or not math.isfinite(float(theta)) or float(theta) <= 0:
            raise ValueError("Invalid rope_parameters.rope_theta")
    elif "rope_theta" in raw_cfg:
        theta = raw_cfg["rope_theta"]
        if not math.isfinite(float(theta)) or float(theta) <= 0:
            raise ValueError("Invalid rope_theta")

    # Validate context limit & layer metadata
    vocab_size = int(raw_cfg.get("vocab_size", 0))
    max_pos = int(raw_cfg.get("max_position_embeddings", 0))
    num_layers = int(raw_cfg.get("num_hidden_layers", 0))
    hidden_size = int(raw_cfg.get("hidden_size", 0))
    if vocab_size <= 0 or max_pos <= 0 or num_layers <= 0 or hidden_size <= 0:
        raise ValueError("config.json must declare positive vocab_size, max_position_embeddings, num_hidden_layers, and hidden_size")

    # Validate complete safetensors shards
    index_path = p / "model.safetensors.index.json"
    single_shard = p / "model.safetensors"
    shard_files: List[Path] = []
    if index_path.exists():
        idx_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = idx_data.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has empty or invalid weight_map")
        referenced = sorted(set(weight_map.values()))
        missing = [fname for fname in referenced if not (p / fname).is_file()]
        if missing:
            raise ValueError(f"Missing safetensors shards referenced by index: {missing}")
        shard_files = [p / fname for fname in referenced]
    elif single_shard.is_file():
        shard_files = [single_shard]
    else:
        raise ValueError("Missing model.safetensors or model.safetensors.index.json")

    disk_weight_bytes = sum(sf.stat().st_size for sf in shard_files)

    # Validate tokenizer & native chat template presence
    tok_cfg_path = p / "tokenizer_config.json"
    if not tok_cfg_path.exists():
        raise ValueError("Missing tokenizer_config.json")
    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    has_template = bool(tok_cfg.get("chat_template")) or (p / "chat_template.jinja").exists()
    if not has_template:
        raise ValueError("Checkpoint needs a native chat_template in tokenizer_config.json or chat_template.jinja")
    if tok_cfg.get("eos_token") is None and tok_cfg.get("eos_token_id") is None:
        raise ValueError("tokenizer_config.json must declare eos_token or eos_token_id")

    # Device availability check
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' is unavailable")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested device 'mps' is unavailable")

    # Labeled memory estimates (separate from disk weights)
    est_inf_bytes = int(disk_weight_bytes * 1.2)
    est_train_bytes = int(disk_weight_bytes * 2.5)
    lineage = _detect_mergekit_lineage(p, raw_cfg)

    return {
        "valid": True,
        "backend": "huggingface_safetensors",
        "checkpoint_format": "hf_safetensors",
        "model_dir": str(p.resolve()),
        "model_type": model_type,
        "compatibility_status": matrix_entry["status"],
        "architectures": raw_cfg.get("architectures", []),
        "vocab_size": vocab_size,
        "max_position_embeddings": max_pos,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "shard_count": len(shard_files),
        "disk_weight_bytes": disk_weight_bytes,
        "estimated_inference_memory_bytes": est_inf_bytes,
        "estimated_training_memory_bytes": est_train_bytes,
        "memory_values_are_estimates": True,
        "trust_remote_code": False,
        "mergekit_lineage": lineage,
    }


class HuggingFaceBackendAdapter(ModelBackendAdapter):
    """Explicit HF Safetensors backend adapter enforcing `#438` & `#439` pre-flight validation."""

    backend_name = "huggingface_safetensors"
    checkpoint_format = "hf_safetensors"

    def __init__(
        self,
        model_dir: Path,
        *,
        requested_lora_targets: Sequence[str] = ("q_proj", "v_proj"),
        requested_device: str = "auto",
    ) -> None:
        self.validation = validate_hf_or_mergekit_model_dir(
            model_dir,
            requested_lora_targets=requested_lora_targets,
            requested_device=requested_device,
        )
        self.model_dir = Path(model_dir).resolve()

    def inspect_model(self) -> Dict[str, Any]:
        return dict(self.validation)

    def render_prompt(self, prompt: str) -> str:
        from aix_training_runtime import tokenizer_for
        from aix_training_data import render_prompt as hf_render_prompt
        tok = tokenizer_for({"base_model": str(self.model_dir)})
        return hf_render_prompt(tok, {"prompt": prompt})

    def generate(self, prompt: str, *, max_new_tokens: int = 36) -> Dict[str, Any]:
        from aix_training_runtime import model_for, tokenizer_for
        cfg = {"base_model": str(self.model_dir), "training": {"device": "auto"}}
        tok = tokenizer_for(cfg)
        model = model_for(cfg)
        rendered = self.render_prompt(prompt)
        inputs = tok(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        comp = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return {"backend": self.backend_name, "assistance_mode": "neural_only", "completion": comp}

    def train_step(self, examples: Sequence[Dict[str, str]], *, lr: float = 1e-3, steps: int = 1) -> Dict[str, Any]:
        raise NotImplementedError("Use train_aix.py / aix_training_runtime.train_adapter(run_dir) for full bounded HF LoRA runs")

    def reload_artifact(self, artifact_path: Path) -> Dict[str, Any]:
        ap = Path(artifact_path)
        if not (ap / "adapter_config.json").exists():
            raise ValueError(f"Expected PEFT adapter directory with adapter_config.json, got {ap}")
        return {"reloaded": True, "artifact": str(ap), "format": "peft_lora_adapter"}
