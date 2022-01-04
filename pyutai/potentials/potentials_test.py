"""
Module that implements automatic test cases for TreePotential class.
"""

import unittest
import numpy as np


class TreeTestCase(unittest.TestCase):
    """
    Test Class for TreePotential class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        array1 = np.array([[1, 1], [2, 2]])
        array2 = np.array([1])

        self.tree1 = Tree.from_array(array1)
        self.tree2 = Tree.from_array(array2)

    def test_from_array(self):

        with self.assertRaise(ValueError):
            Tree.from_array(np.array([]))


if __name__ == '__main__':
    unittest.main()
