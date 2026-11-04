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

CHAR_VOCAB: List[str] = list(
    "0123456789"
    "+-*/=%()[]{}<>!=^&|~_.,:;?@#$"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " \n\t"
)

# Whole-word tokens for English math word problems, calculus, 2x2 systems & Python code traces
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
]


class AIXZeroTokenizer:
    """Deterministic hybrid word + atomic-character tokenizer built from first principles."""

    def __init__(self, use_word_vocab: bool = True) -> None:
        base_tokens = list(SPECIAL_TOKENS) + list(CHAR_VOCAB)
        if use_word_vocab:
            base_tokens.extend(WORD_VOCAB)
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
