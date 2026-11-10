"""From-scratch Hybrid Word + Atomic-Digit Tokenizer for AIX-Zero v4.

Design:
1. Dedicated structural reasoning tags:
   <pad>, <bos>, <eos>, <think>, </think>, <verify>, </verify>, <ans>, </ans>
2. Atomic single-character tokens for all digits (0-9), operators (+-*/=%^><), and ASCII characters
   so multi-digit numbers and algebraic expressions never split inconsistently.
3. Whole-word tokens (WORD_VOCAB) for English math word problems, symbolic calculus, 2x2 systems,
   and Python code execution prompts so natural-language/code prompts compress into short token
   sequences while keeping numbers atomic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence


SPECIAL_TOKENS: List[str] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<think>",
    "</think>",
    "<verify>",
    "</verify>",
    "<ans>",
    "</ans>",
]

CHAR_VOCAB: List[str] = list(
    "0123456789"
    "+-*/=%()[]{}<>!=^&|~_.,:;?@#$"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " \n\t"
)

# Whole-word tokens for English math word problems, calculus, 2x2 systems, code traces & v6-v8 domains
WORD_VOCAB: List[str] = [
    "Q:Word:",
    "Q:Calc:",
    "d/dx",
    " packs ",
    " of ",
    " and ",
    " buys ",
    " loses ",
    " more ",
    " left=",
    " total=",
    "Sam ",
    "Mia ",
    "Leo ",
    "has ",
    "Q:Sys:",
    "Q:Code:",
    "def f(x):return ",
    "Q:Quad:",
    "Q:Phys:",
    "Q:Geom:",
    "Q:Seq:",
    "Q:Chain3:",
    "Q:Elec:",
    "Q:Fin:",
    "Q:Thermo:",
    "Q:Comb:",
    "Q:Prob:",
    "Q:BridgeET:",
    "Q:BridgeKW:",
    "Q:Chain4:",
]


class AIXZeroTokenizer:
    """Deterministic hybrid word + atomic-character tokenizer built from first principles."""

    def __init__(
        self,
        use_word_vocab: bool = True,
        extra_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        base_tokens = list(SPECIAL_TOKENS) + list(CHAR_VOCAB)
        if use_word_vocab:
            base_tokens.extend(WORD_VOCAB)
        if extra_tokens:
            for tok in extra_tokens:
                if not isinstance(tok, str) or not tok:
                    raise ValueError("Tokenizer tokens must be nonempty strings")
                if tok in base_tokens:
                    raise ValueError(f"Duplicate token in vocabulary: {tok!r}")
                base_tokens.append(tok)
        self.id_to_token: List[str] = base_tokens
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.use_word_vocab = use_word_vocab

        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.think_id = self.token_to_id["<think>"]
        self.end_think_id = self.token_to_id["</think>"]
        self.ans_id = self.token_to_id["<ans>"]
        self.end_ans_id = self.token_to_id["</ans>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def prefix_sha256(self, length: Optional[int] = None) -> str:
        """Compute canonical SHA-256 over the ordered token list (or a prefix of length `length`)."""
        n = self.vocab_size if length is None else int(length)
        if n < 0 or n > self.vocab_size:
            raise ValueError("Vocabulary prefix length out of range")
        canonical = json.dumps(self.id_to_token[:n], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def vocab_sha256(self) -> str:
        """Canonical SHA-256 digest of the ordered vocabulary."""
        return self.prefix_sha256(self.vocab_size)

    def append_token(self, token: str) -> int:
        """Explicitly append a new token at the end of the vocabulary, preserving all existing IDs."""
        if not isinstance(token, str) or not token:
            raise ValueError("Token must be a nonempty string")
        if token in self.token_to_id:
            return self.token_to_id[token]
        new_id = len(self.id_to_token)
        self.id_to_token.append(token)
        self.token_to_id[token] = new_id
        return new_id

    def to_metadata(self) -> Dict[str, Any]:
        """Export ordered vocabulary identity metadata for binding to checkpoints and manifests."""
        return {
            "schema_version": 1,
            "tokenizer_class": "AIXZeroTokenizer",
            "use_word_vocab": self.use_word_vocab,
            "vocab_size": self.vocab_size,
            "vocab_sha256": self.vocab_sha256,
            "id_to_token": list(self.id_to_token),
        }

    def verify_checkpoint_compatibility(
        self,
        checkpoint_vocab_size: int,
        expected_metadata: Optional[Dict[str, Any]] = None,
        *,
        allow_append_only_superset: bool = False,
    ) -> None:
        """Verify that this tokenizer's ordered vocabulary matches a checkpoint's expected identity.

        Raises ValueError if:
        - `checkpoint_vocab_size` exceeds `self.vocab_size` (would require random embedding expansion),
        - `checkpoint_vocab_size != self.vocab_size` when `allow_append_only_superset=False`,
        - `expected_metadata` specifies a token ordering or `vocab_sha256` that does not match.
        """
        if checkpoint_vocab_size <= 0:
            raise ValueError("checkpoint_vocab_size must be positive")
        if self.vocab_size < checkpoint_vocab_size:
            raise ValueError("Tokenizer is smaller than checkpoint vocabulary")
        if not allow_append_only_superset and self.vocab_size != checkpoint_vocab_size:
            raise ValueError(
                f"Tokenizer vocab_size ({self.vocab_size}) does not match checkpoint ({checkpoint_vocab_size})"
            )
        if expected_metadata is not None:
            meta_tokens = expected_metadata.get("id_to_token")
            meta_size = expected_metadata.get("vocab_size", checkpoint_vocab_size)
            meta_sha = expected_metadata.get("vocab_sha256")
            if meta_size != checkpoint_vocab_size:
                raise ValueError("Metadata vocab_size does not match checkpoint_vocab_size")
            if meta_tokens is not None:
                if not isinstance(meta_tokens, list) or len(meta_tokens) != checkpoint_vocab_size:
                    raise ValueError("Metadata id_to_token length mismatch")
                if self.id_to_token[:checkpoint_vocab_size] != meta_tokens:
                    raise ValueError("Tokenizer ordered vocabulary does not match checkpoint metadata")
            if meta_sha is not None:
                if self.prefix_sha256(checkpoint_vocab_size) != meta_sha:
                    raise ValueError("Tokenizer vocab_sha256 does not match checkpoint metadata")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        i = 0
        n = len(text)
        while i < n:
            matched = False
            if text[i] == "<":
                for sp in SPECIAL_TOKENS:
                    if text.startswith(sp, i):
                        ids.append(self.token_to_id[sp])
                        i += len(sp)
                        matched = True
                        break
            if not matched and self.use_word_vocab:
                for w in WORD_VOCAB:
                    if text.startswith(w, i):
                        ids.append(self.token_to_id[w])
                        i += len(w)
                        matched = True
                        break
            if not matched:
                ch = text[i]
                if ch in self.token_to_id:
                    ids.append(self.token_to_id[ch])
                i += 1

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        pieces: List[str] = []
        for idx in ids:
            if 0 <= idx < len(self.id_to_token):
                tok = self.id_to_token[idx]
                if skip_special and tok in ("<pad>", "<bos>", "<eos>"):
                    continue
                pieces.append(tok)
        return "".join(pieces)


def build_v11_tokenizer() -> AIXZeroTokenizer:
    """Return the exact 134-token ordered tokenizer bound to `aix_zero_v11_kip.safetensors`."""
    return AIXZeroTokenizer(use_word_vocab=True)

