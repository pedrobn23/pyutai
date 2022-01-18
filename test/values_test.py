"""
Module that implements automatic test cases for TreePotential class.
"""

import itertools
import unittest
import numpy as np

from pyutai import values, nodes


class TreeTestCase(unittest.TestCase):
    """
    Test Class for values.Tree class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [
            np.array([1]),
            np.array([[1, 6], [2, 2]]),
            np.array([[[1, 90876], [1, 7]], [[2, 34], [3, 23]]])
        ]

        self.trees = [values.Tree.from_array(arr) for arr in self.arrays]
        self.maxDiff = 1000

    def test_execptions_from_array(self):
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([]))

    def _check_trees(self, arrays, trees):
        for arr, tree in zip(arrays, trees):
            for element in tree:
                self.assertEqual(arr[element.state], element.value)

    def test_access(self):
        self._check_trees(self.arrays, self.trees)

    def test_restraint(self):
        for arr, tree in zip(self.arrays, self.trees):
            if arr.shape[0] > 1:  # otherwise there is little to restrain
                tree.restrain(0, 0)

                state = [i - 1 for i in arr.shape]
                restrained_state = [i - 1 for i in arr.shape]
                restrained_state[0] = 0

                self.assertEqual(arr[tuple(restrained_state)],
                                 tree.access(state))
                tree.unrestrain(0)

        # to check that variables are properly unrestained
        self._check_trees(self.arrays, self.trees)

    def test_prune(self):
        arr = np.array([[1, 6], [2, 2]])
        tree = values.Tree.from_array(arr)
        tree.prune()

        pruned_repr = '''Tree(root=<class 'pyutai.nodes.BranchNode'>(0, [<class 'pyutai.nodes.BranchNode'>(1, [<class 'pyutai.nodes.LeafNode'>(1), <class 'pyutai.nodes.LeafNode'>(6)]), <class 'pyutai.nodes.LeafNode'>(2)]), cardinality=[2, 2], restraints={})'''

        self.assertEqual(repr(tree), pruned_repr)


if __name__ == '__main__':
    unittest.main()
