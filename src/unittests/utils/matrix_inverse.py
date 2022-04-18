import unittest

import numpy as np
from numpy.linalg import LinAlgError

from src.utils import matrix_inverse


class TestMatrixInverse(unittest.TestCase):
    def test_inverse(self):
        a = np.eye(10)
        a_inv = np.eye(10)
        np.testing.assert_array_almost_equal(matrix_inverse(a), a_inv)

        a = np.diag([1, 2, 3, 4, 5])
        a_inv = np.linalg.inv(np.diag([1, 2, 3, 4, 5]))
        np.testing.assert_array_almost_equal(matrix_inverse(a), a_inv)

        a = np.random.random(size=(10, 10))
        a = a.T @ a + np.eye(10)
        a_inv = np.linalg.inv(a)
        np.testing.assert_array_almost_equal(matrix_inverse(a), a_inv)

        a = np.zeros((10, 10))
        with self.assertRaises(LinAlgError):
            matrix_inverse(a)
