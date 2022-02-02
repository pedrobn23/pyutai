"""
Module that implements automatic test cases for Node classes.
"""
import itertools
import unittest
import numpy as np

from pyutai import values, nodes


class NodesTestCase(unittest.TestCase):
    """
    Test Class for Node classes in nodes.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [
            np.array([[1, 6, 3], [2, 2, 2]]),
            np.array([[[1, 90876], [1, 7], [0, 0]], [[2, 34], [3, 23], [0, 0]]])
        ]
        self.variables = [['A', 'B'], ['A', 'B', 'C']]

        self.cardinalities = {'A': 2, 'B': 3, 'C': 2}
        self.branch_nodes = [
            values.Tree.from_array(arr, variables, self.cardinalities).root
            for arr, variables in zip(self.arrays, self.variables)
        ]

        self.n_leaf_nodes = 5
        self.leaf_nodes = [nodes.LeafNode(i) for i in range(self.n_leaf_nodes)]

        self.all_nodes = self.branch_nodes + self.leaf_nodes

    def test_equality(self):
        other_branch_nodes = [
            values.Tree.from_array(arr, variables, self.cardinalities).root
            for arr, variables in zip(self.arrays, self.variables)
        ]
        for nodeA, nodeB in zip(self.branch_nodes, other_branch_nodes):
            self.assertEqual(nodeA, nodeB)

        other_leaf_nodes = [nodes.LeafNode(i) for i in range(self.n_leaf_nodes)]
        for nodeA, nodeB in zip(self.leaf_nodes, other_leaf_nodes):
            self.assertEqual(nodeA, nodeB)

        for nodeA, nodeB in itertools.permutations(self.all_nodes, 2):
            self.assertNotEqual(nodeA, nodeB)


class TableNodeTestCase(unittest.TestCase):
    """
    Test Class for TableNode classes in nodes.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxDiff = 1000

    def test_sum(self):
        tablea = np.array([[1, 2], [3, 4]])
        tableb = np.array([[1, 1], [1, 9]])
        tablec = np.array([[[2, 2], [3, 11]], [[4, 4], [5, 13]]])

        nodea = nodes.TableNode(values=tablea, variables=['A', 'B'])
        nodeb = nodes.TableNode(values=tableb, variables=['B', 'C'])
        nodec = nodes.TableNode(values=tablec, variables=['A', 'B', 'C'])

        self.assertEqual(nodea + nodeb, nodec)

    def test_prod(self):
        tablea = np.array([[1, 2], [3, 4]])
        tableb = np.array([[1, 1], [1, 9]])
        tablec = np.array([[[1, 1], [2, 18]], [[3, 3], [4, 36]]])

        nodea = nodes.TableNode(values=tablea, variables=['A', 'B'])
        nodeb = nodes.TableNode(values=tableb, variables=['B', 'C'])
        nodec = nodes.TableNode(values=tablec, variables=['A', 'B', 'C'])

        print(nodea * nodeb)
        print(nodec)
        self.assertEqual(nodea * nodeb, nodec)


if __name__ == '__main__':
    unittest.main()
