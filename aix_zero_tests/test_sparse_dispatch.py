"""Dense-reference equivalence, optimizer semantics, and executed token coverage."""
import copy
from pathlib import Path
import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, MixtureOfReasoningExperts, profile_moe_dispatch_breakdown
from aix_zero.benchmark_baseline import load_v11


class SparseDispatchTests(unittest.TestCase):
    def test_outputs_input_and_parameter_gradients_match(self):
        for shared in (False, True):
            for experts, top_k in ((1, 1), (4, 1), (4, 2), (4, 4), (8, 2)):
                with self.subTest(shared=shared, experts=experts, top_k=top_k):
                    torch.manual_seed(4)
                    dense = MixtureOfReasoningExperts(AIXZeroConfig(
                        d_model=16, d_ff=64, n_experts=experts,
                        moe_top_k=top_k, use_shared_expert=shared)).double()
                    sparse = copy.deepcopy(dense)
                    sparse.dispatch_mode = 'sparse'
                    # Noncontiguous batch/sequence input.
                    x = torch.randn(5, 2, 16, dtype=torch.float64).transpose(0, 1).requires_grad_()
                    y = x.detach().clone().requires_grad_()
                    expected, aux = dense(x)
                    actual, sparse_aux = sparse(y)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                    torch.testing.assert_close(sparse_aux, aux, rtol=0, atol=0)
                    (expected.square().sum() + aux).backward()
                    (actual.square().sum() + sparse_aux).backward()
                    torch.testing.assert_close(y.grad, x.grad, rtol=1e-9, atol=1e-11)
                    for (name, p), (_, q) in zip(dense.named_parameters(), sparse.named_parameters()):
                        self.assertEqual(p.grad is None, q.grad is None, name)
                        torch.testing.assert_close(q.grad, p.grad, rtol=1e-9, atol=1e-11)

    def test_only_selected_tokens_executed_and_adamw_matches_for_idle_experts(self):
        torch.manual_seed(3)
        dense = MixtureOfReasoningExperts(AIXZeroConfig(
            d_model=16, d_ff=64, n_experts=8, moe_top_k=2)).double()
        # Positive inputs always select experts 0 and 1; others receive no tokens.
        with torch.no_grad():
            dense.router.weight.fill_(-1)
            dense.router.weight[0].fill_(1)
            dense.router.weight[1].fill_(.5)
        sparse = copy.deepcopy(dense)
        sparse.dispatch_mode = 'sparse'
        counts = [0] * 8
        handles = []
        for i, expert in enumerate(sparse.experts):
            def hook(module, args, index=i):
                counts[index] += args[0].shape[0]
            handles.append(expert.register_forward_pre_hook(hook))
        self.addCleanup(lambda: [h.remove() for h in handles])
        x = torch.rand(2, 3, 16, dtype=torch.float64)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.1) for m in (dense, sparse)]
        for _ in range(2):
            for model, optimizer in zip((dense, sparse), optimizers):
                optimizer.zero_grad(set_to_none=True)
                out, aux = model(x)
                (out.square().mean() + aux).backward()
                optimizer.step()
        self.assertEqual(counts, [12, 12, 0, 0, 0, 0, 0, 0])
        for (name, p), (_, q) in zip(dense.named_parameters(), sparse.named_parameters()):
            torch.testing.assert_close(q, p, rtol=1e-9, atol=1e-11, msg=name)

    def test_real_v11_forward_and_greedy_generation_match(self):
        torch.set_num_threads(1)
        checkpoint = Path(__file__).resolve().parents[1] / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        dense, tokenizer = load_v11(checkpoint)
        sparse = copy.deepcopy(dense)
        sparse.set_expert_dispatch('sparse')
        self.assertEqual(set(dense.state_dict()), set(sparse.state_dict()))
        for prompt in ('Q:(3*4)+5=', 'Q:Chain4:3*4+5-2+6='):
            ids = tokenizer.encode(prompt)
            with torch.no_grad():
                a, b = dense(torch.tensor([ids])), sparse(torch.tensor([ids]))
            for key in a:
                torch.testing.assert_close(b[key], a[key], rtol=1e-4, atol=2e-5)
            for assisted in (False, True):
                args = dict(eos_id=tokenizer.eos_id, max_new_tokens=40,
                            tokenizer=tokenizer, inflight_alu=assisted)
                self.assertEqual(dense.generate_verified(ids, **args)[0], sparse.generate_verified(ids, **args)[0])
        with self.assertRaises(ValueError):
            sparse.set_expert_dispatch('invalid')

    def test_float32_and_float64_equivalence_and_unsupported_dtype_rejection(self):
        for dtype, rtol, atol in ((torch.float32, 1e-5, 1e-6), (torch.float64, 1e-10, 1e-12)):
            torch.manual_seed(7)
            dense = MixtureOfReasoningExperts(
                AIXZeroConfig(d_model=16, d_ff=64, n_experts=4, moe_top_k=2, use_shared_expert=True)
            ).to(dtype=dtype)
            sparse = copy.deepcopy(dense)
            sparse.dispatch_mode = 'sparse'
            x = torch.randn(3, 4, 16, dtype=dtype, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            out_d, aux_d = dense(x)
            out_s, aux_s = sparse(y)
            torch.testing.assert_close(out_s, out_d, rtol=rtol, atol=atol)
            (out_d.sum() + aux_d).backward()
            (out_s.sum() + aux_s).backward()
            torch.testing.assert_close(y.grad, x.grad, rtol=rtol, atol=atol)

        # Unsupported precision mode (e.g. float16 on CPU MoRE) fails explicitly (`EXPERT-09`)
        half_moe = MixtureOfReasoningExperts(AIXZeroConfig(d_model=16, d_ff=64, n_experts=4, moe_top_k=2)).half()
        with self.assertRaisesRegex(ValueError, "Unsupported precision dtype"):
            half_moe(torch.randn(2, 3, 16, dtype=torch.float16))

    def test_expert_10_dispatch_profiling_separates_microkernel_from_full_forward(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer = load_v11(checkpoint)
        ids = torch.tensor([tokenizer.encode('Q:(3*4)+5=')])
        prof = profile_moe_dispatch_breakdown(model, ids, warmup_runs=1, measured_runs=2)
        self.assertEqual(prof['default_dispatch_mode'], 'dense')
        self.assertIn('moe_layer_dense_ms', prof['microkernel_ms'])
        self.assertIn('moe_layer_sparse_ms', prof['microkernel_ms'])
        self.assertIn('dense_ms', prof['full_forward_ms'])
        self.assertIn('sparse_ms', prof['full_forward_ms'])


if __name__ == '__main__':
    unittest.main()
