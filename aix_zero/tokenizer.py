"""From-scratch symbolic & character-level tokenizer for AIX-Zero.

Unlike BPE tokenizers that merge arbitrary multi-digit numbers into inconsistent chunks,
AIXZeroTokenizer assigns dedicated atomic token IDs to structural reasoning tags
(<bos>, <eos>, <pad>, <think>, </think>, <verify>, </verify>, <ans>, </ans>)
and single-character tokens to every digit, operator, variable, and ASCII symbol.
"""

from __future__ import annotations

from typing import List


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

# Atomic printable vocabulary: digits, math operators, letters, punctuation, whitespace
CHAR_VOCAB: List[str] = list(
    "0123456789"
    "+-*/=%()[]{}<>!=^&|~_.,:;?@#$"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " \n\t"
)


class AIXZeroTokenizer:
    """Deterministic atomic tokenizer built from first principles."""

    def __init__(self) -> None:
        self.id_to_token: List[str] = list(SPECIAL_TOKENS) + list(CHAR_VOCAB)
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
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
            matched_special = False
            if text[i] == "<":
                for sp in SPECIAL_TOKENS:
                    if text.startswith(sp, i):
                        ids.append(self.token_to_id[sp])
                        i += len(sp)
                        matched_special = True
                        break
            if not matched_special:
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
