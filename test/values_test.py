"""
Module that implements automatic test cases for TreePotential class.
"""

import itertools
import unittest
import numpy as np

from pyutai import values


class NodesTestCase(unittest.TestCase):
    """
    Test Class for Node classes in values.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [
            np.array([[1, 6], [2, 2]]),
            np.array([[[1, 90876], [1, 7]], [[2, 34], [3, 23]]])
        ]

        self.branch_nodes = [
            values.Tree.from_array(arr).root for arr in self.arrays
        ]

        self.n_leaf_nodes = 5
        self.leaf_nodes = [values.LeafNode(i) for i in range(self.n_leaf_nodes)]

        self.all_nodes = self.branch_nodes + self.leaf_nodes

    def testEquality(self):
        other_branch_nodes = [
            values.Tree.from_array(arr).root for arr in self.arrays
        ]
        for nodeA, nodeB in zip(self.branch_nodes, other_branch_nodes):
            self.assertEqual(nodeA, nodeB)

        other_leaf_nodes = [
            values.LeafNode(i) for i in range(self.n_leaf_nodes)
        ]
        for nodeA, nodeB in zip(self.leaf_nodes, other_leaf_nodes):
            self.assertEqual(nodeA, nodeB)

        for nodeA, nodeB in itertools.permutations(self.all_nodes, 2):
            self.assertNotEqual(nodeA, nodeB)


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

    def test_execptions_from_array(self):
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([]))

    def _check_trees(self, arrays, trees):
        for arr, tree in zip(arrays, trees):
            for states, value in tree:
                self.assertEqual(arr[states], value)

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

        pruned_repr = '''Tree(root=<class 'pyutai.values.BranchNode'>(0, [<class 'pyutai.values.BranchNode'>(1, [<class 'pyutai.values.LeafNode'>(1), <class 'pyutai.values.LeafNode'>(6)]), <class 'pyutai.values.LeafNode'>(2)]), cardinality=(2, 2), restraints={})'''

        self.assertEqual(repr(tree), pruned_repr)


if __name__ == '__main__':
    unittest.main()
