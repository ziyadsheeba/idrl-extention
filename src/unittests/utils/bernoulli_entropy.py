import unittest

import numpy as np

from src.utils import bernoulli_entropy


class TestBernoulliEntropy(unittest.TestCase):
    def test_bernoulli_entropy(self):
        p = 0.5
        self.assertAlmostEqual(bernoulli_entropy(p), np.log(2))

        with self.assertRaises(ValueError):
            bernoulli_entropy(-1)
            bernoulli_entropy(5)
