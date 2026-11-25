import unittest
from aix_zero.train_ai_coding import curriculum, interpret, correct


class CodingTests(unittest.TestCase):
    def test_disjoint_and_reference_correct(self):
        train, dev = curriculum('train'), curriculum('development')
        self.assertFalse({r['prompt'] for r in train} & {r['prompt'] for r in dev})
        for row in train+dev:
            self.assertTrue(correct(row['answer'], row['answer']))

    def test_functional_equivalence_and_wrong_formulas(self):
        self.assertTrue(correct('w-g*lr*3', 'w-3*lr*g'))
        self.assertFalse(correct('w+3*lr*g', 'w-3*lr*g'))
        self.assertFalse(correct('base+3*r*a*b', 'base+3/r*a*b'))
        self.assertFalse(correct('0', 'x+b'))

    def test_reject_executable_and_unbounded_syntax(self):
        for source in ('__import__("os")', 'x.__class__', '[x for x in []]',
                       '2**100000', '1/0', '1e999', 'True', 'unknown', 'x'*129):
            self.assertFalse(correct(source, 'x'))


if __name__ == '__main__':
    unittest.main()
