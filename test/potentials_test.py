"""
Module that implements automatic test cases for TreePotential class.
"""

import unittest
import numpy as np

from pyutai.values import Tree


class TreeTestCase(unittest.TestCase):
    """
    Test Class for TreePotential class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [
            np.array([1]),
            np.array([[1, 1], [2, 2]]),
            np.array([[[1, 1], [1, 7]], [[2, 34], [3, 23]]])
        ]

        self.trees = [values.Tree.from_array(arr) for arr in self.arrays]

    def test_from_array(self):

        with self.assertRaise(ValueError):
            Tree.from_array(np.array([]))


if __name__ == '__main__':
    unittest.main()
