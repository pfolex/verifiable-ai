import unittest

from aix_zero.benchmark_capacity_expert import build_cases


class CapacityExpertBenchmarkTests(unittest.TestCase):
    def test_splits_hold_out_operand_pairs_and_test_layout(self):
        splits = build_cases(20261111)
        self.assertEqual(splits, build_cases(20261111))
        seen = set()
        for split, rows in splits.items():
            groups = {row['group'] for row in rows}
            self.assertTrue(seen.isdisjoint(groups))
            seen.update(groups)
            self.assertEqual(
                {row['layout'] for row in rows},
                {'standard', 'novel'} if split == 'test' else {'standard'},
            )
            for row in rows:
                top_k, width = map(int, row['group'].split(':'))
                self.assertEqual(row['gold'], str(top_k * width))
