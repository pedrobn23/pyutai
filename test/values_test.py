"""
Module that implements automatic test cases for TreePotential class.
"""

import itertools
import unittest
import numpy as np

from pyutai import values


class TreeTestCase(unittest.TestCase):
    """
    Test Class for values.Tree class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [
            np.array([1]),
            np.array([[1, 1], [2, 2]]),
            np.array([[[1, 1], [1, 7]], [[2, 34], [3, 23]]])
        ]

        self.trees = [values.Tree.from_array(arr) for arr in self.arrays]

    def test_execptions_from_array(self):
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([]))

    def test_access(self):
        for arr, tree in zip(self.arrays, self.trees):
            for var in itertools.product(*[range(var) for var in arr.shape]):
                self.assertEqual(arr[var], tree.access(var))


if __name__ == '__main__':
    unittest.main()
