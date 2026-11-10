"""Unit tests for ordered tokenizer identity and checkpoint binding (BASE-03)."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.benchmark_baseline import load_v11
from aix_zero.tokenizer import AIXZeroTokenizer, build_v11_tokenizer

ROOT = Path(__file__).resolve().parents[1]


class TokenizerIdentityTests(unittest.TestCase):
    def test_v11_tokenizer_metadata_is_deterministic(self):
        tok1 = build_v11_tokenizer()
        tok2 = AIXZeroTokenizer(use_word_vocab=True)
        self.assertEqual(tok1.vocab_size, 134)
        self.assertEqual(tok1.id_to_token, tok2.id_to_token)
        self.assertEqual(tok1.vocab_sha256, tok2.vocab_sha256)
        meta = tok1.to_metadata()
        self.assertEqual(meta["vocab_size"], 134)
        self.assertEqual(meta["vocab_sha256"], tok1.vocab_sha256)
        self.assertEqual(meta["id_to_token"], tok1.id_to_token)

    def test_reordered_or_altered_vocab_with_same_size_is_rejected(self):
        tok = build_v11_tokenizer()
        meta = tok.to_metadata()

        # Swap two tokens while keeping vocab_size == 134 identical
        reordered_meta = copy.deepcopy(meta)
        reordered_meta["id_to_token"][-1], reordered_meta["id_to_token"][-2] = (
            reordered_meta["id_to_token"][-2],
            reordered_meta["id_to_token"][-1],
        )
        with self.assertRaisesRegex(ValueError, "ordered vocabulary does not match"):
            tok.verify_checkpoint_compatibility(134, reordered_meta)

        # Mutate vocab_sha256 while keeping vocab_size == 134 identical
        tampered_sha_meta = copy.deepcopy(meta)
        tampered_sha_meta["vocab_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "vocab_sha256 does not match"):
            tok.verify_checkpoint_compatibility(134, tampered_sha_meta)

    def test_smaller_tokenizer_is_rejected_on_checkpoint_load(self):
        small_tok = AIXZeroTokenizer(use_word_vocab=False)
        self.assertLess(small_tok.vocab_size, 134)
        with self.assertRaisesRegex(ValueError, "smaller than checkpoint vocabulary"):
            small_tok.verify_checkpoint_compatibility(134)

    def test_append_only_expansion_preserves_prefix_ids_and_weights(self):
        base_tok = build_v11_tokenizer()
        base_meta = base_tok.to_metadata()
        expanded_tok = AIXZeroTokenizer(
            use_word_vocab=True, extra_tokens=["Q:Bayes:", "Q:Causal:"]
        )
        self.assertEqual(expanded_tok.vocab_size, 136)
        self.assertEqual(expanded_tok.id_to_token[:134], base_tok.id_to_token)
        self.assertEqual(expanded_tok.prefix_sha256(134), base_tok.vocab_sha256)

        # Strict mode rejects mismatched size, while append-only superset mode accepts matching prefix
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            expanded_tok.verify_checkpoint_compatibility(
                134, base_meta, allow_append_only_superset=False
            )
        expanded_tok.verify_checkpoint_compatibility(
            134, base_meta, allow_append_only_superset=True
        )

        # Verify resize_token_embeddings is explicit and preserves all existing 134 token embeddings
        cfg = AIXZeroConfig(vocab_size=134, d_model=32, n_heads=2, n_kv_heads=1, d_ff=64)
        model = AIXZeroModel(cfg)
        before_emb = model.token_emb.weight.detach().clone()
        before_lm = model.lm_head.weight.detach().clone()
        model.resize_token_embeddings(expanded_tok.vocab_size)
        self.assertEqual(model.config.vocab_size, 136)
        self.assertTrue(torch.equal(model.token_emb.weight[:134], before_emb))
        self.assertTrue(torch.equal(model.lm_head.weight[:134], before_lm))

    def test_load_v11_verifies_expected_tokenizer_metadata(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        tok = build_v11_tokenizer()
        model, loaded_tok = load_v11(ckpt, expected_tokenizer_metadata=tok.to_metadata())
        self.assertEqual(model.config.vocab_size, 134)
        self.assertEqual(loaded_tok.vocab_sha256, tok.vocab_sha256)

        bad_meta = tok.to_metadata()
        bad_meta["vocab_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "vocab_sha256 does not match"):
            load_v11(ckpt, expected_tokenizer_metadata=bad_meta)


if __name__ == "__main__":
    unittest.main()
