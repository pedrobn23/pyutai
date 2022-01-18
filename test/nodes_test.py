"""
Module that implements automatic test cases for Node classes.
"""
import itertools
import unittest
import numpy as np

from pyutai import values, nodes


class NodesTestCase(unittest.TestCase):
    """
    Test Class for Node classes in values.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [
            np.array([[1, 6], [2, 2]]),
            np.array([[[1, 90876], [1, 7], [0,0]], [[2, 34], [3, 23],[0,0]]])
        ]
        self.variables = [['A','B'], ['A','B', 'C']]

        self.cardinalities = {'A':2, 'B':2, 'C': 3}
        self.branch_nodes = [
            values.Tree.from_array(arr, variables, self.cardinalities).root for arr, variables in zip(self.arrays, self.variables)
        ]

        self.n_leaf_nodes = 5
        self.leaf_nodes = [nodes.LeafNode(i) for i in range(self.n_leaf_nodes)]

        self.all_nodes = self.branch_nodes + self.leaf_nodes

    def testEquality(self):
        other_branch_nodes = [
            values.Tree.from_array(arr).root for arr in self.arrays
        ]
        for nodeA, nodeB in zip(self.branch_nodes, other_branch_nodes):
            self.assertEqual(nodeA, nodeB)

        other_leaf_nodes = [nodes.LeafNode(i) for i in range(self.n_leaf_nodes)]
        for nodeA, nodeB in zip(self.leaf_nodes, other_leaf_nodes):
            self.assertEqual(nodeA, nodeB)

        for nodeA, nodeB in itertools.permutations(self.all_nodes, 2):
            self.assertNotEqual(nodeA, nodeB)


if __name__ == '__main__':
    unittest.main()
