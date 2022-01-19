"""
Module that implements automatic test cases for TreePotential class.
"""

import copy
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
            np.array([[1, 6, 3], [2, 2, 2]]),
            np.array([[[1, 90876], [1, 7], [0, 0]], [[2, 34], [3, 23], [0, 0]]])
        ]
        self.variables = [['A', 'B'], ['A', 'B', 'C']]

        self.cardinalities = {'A': 2, 'B': 3, 'C': 2}
        self.trees = [
            values.Tree.from_array(arr, variables, self.cardinalities)
            for arr, variables in zip(self.arrays, self.variables)
        ]

        self.maxDiff = 1000

    def test_deepcopy_and_equality(self):
        for tree in self.trees:
            other = copy.deepcopy(tree)
            self.assertEqual(other, tree)
            self.assertNotEqual(id(other), id(tree))

    def test_exceptions_from_array(self):
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([]), [], {})
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([1]), ['A'], {'A': 2})

    def _check_trees(self, arrays, trees):

        for arr, tree in zip(arrays, trees):
            # adapter from IndexType to numpy Index-Tuple
            data_accessor = lambda x: arr.item(
                tuple(x[var] for var in tree.variables))

            for element in tree:
                self.assertEqual(data_accessor(element.state), element.value)

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

        tree = values.Tree.from_array(arr, ['0', '1'], {'0': 2, '1': 2})
        print(tree)
        tree.prune()

        print(tree)

        pruned_repr = '''Tree(root=<class 'pyutai.nodes.BranchNode'>('0', [<class 'pyutai.nodes.BranchNode'>('1', [<class 'pyutai.nodes.LeafNode'>(1), <class 'pyutai.nodes.LeafNode'>(6)]), <class 'pyutai.nodes.LeafNode'>(2)]), variables=['0', '1'], cardinalities={'0': 2, '1': 2})'''

        self.assertEqual(repr(tree), pruned_repr)


if __name__ == '__main__':
    unittest.main()
