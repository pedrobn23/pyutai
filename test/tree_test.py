"""
Module that implements automatic test cases for TreePotential class.
"""

import copy
import itertools
import unittest
import numpy as np

from pyutai import values, nodes


class TreeTestCase():
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
            self.tree_creation(arr, variables, self.cardinalities)
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
            self.tree_creation(np.array([]), [], {})
        with self.assertRaises(ValueError):
            self.tree_creation(np.array([1]), ['A'], {'A': 2})

    def test_access(self):
        for arr, tree, variables in zip(self.arrays, self.trees,
                                        self.variables):
            # adapter from IndexType to numpy Index-Tuple
            data_accessor = lambda x: arr.item(
                tuple(x[var] for var in variables))

            for element in tree:
                self.assertEqual(data_accessor(element.state), element.value)

    def test_restrict(self):
        for arr, tree, variables in zip(self.arrays, self.trees,
                                        self.variables):
            if arr.shape[0] > 1:  # otherwise there is little to restrict
                restricted_tree = tree.restrict({'A': 0})

                state = {var: self.cardinalities[var] - 1 for var in variables}
                restricted_state = [
                    self.cardinalities[var] - 1 for var in variables
                ]
                restricted_state[0] = 0
                self.assertEqual(arr[tuple(restricted_state)],
                                 restricted_tree.access(state))

    def test_copy(self):
        arr = np.array([[1, 6], [2, 2]])

        tree = self.tree_creation(arr, ['0', '1'], {'0': 2, '1': 2})
        tree2 = tree.copy()
        
        self.assertEqual(tree, tree2)
        
    def test_prune(self):
        arr = np.array([[1, 6], [2, 2]])

        tree = self.tree_creation(arr, ['0', '1'], {'0': 2, '1': 2})
        
        self.assertEqual(tree.root.size(), 7)  # Complete node 4 + 2 + 1

        tree.prune()

        self.assertEqual(tree.root.size(), 5)  # prune two leaves

        tree.unprune()

        print(tree)
        
    def test_product(self):
        card = {'A': 2, 'B': 2, 'C': 2}

        arr1 = np.array([[1, 2], [2, 3]])
        tree1 = self.tree_creation(arr1, ['A', 'B'], card)

        arr2 = np.array([[4, 5], [0, 0]])
        tree2 = self.tree_creation(arr2, ['B', 'C'], card)

        arr3 = np.array([[[4, 5], [0, 0]], [[8, 10], [0, 0]]])
        tree3 = self.tree_creation(arr3, ['A', 'B', 'C'], card)

        tree4 = tree1.product(tree2)

        self.assertEqual(tree4, tree3)

    def test_sum(self):
        card = {'A': 2, 'B': 2}

        arrs = [
            np.array([[1, 2], [2, 3]]),
            np.array([[4, 5], [0, 0]]),
            np.array([[1, 1], [1, 1]])
        ]

        trees = [self.tree_creation(arr, ['A', 'B'], card) for arr in arrs]

        for zip1, zip2 in itertools.permutations(zip(arrs, trees), 2):
            arr1, tree1 = zip1
            arr2, tree2 = zip2
            arr3 = arr1 + arr2
            tree3 = tree1 + tree2
            tree4 = tree2.sum(tree1)

            self.assertEqual(self.tree_creation(arr3, ['A', 'B'], card),
                             tree3)
            self.assertEqual(self.tree_creation(arr3, ['A', 'B'], card),
                             tree4)

    def test_marginalize(self):
        pass


class StandardTestCase(TreeTestCase, unittest.TestCase):
    """
    Test Class for values.Tree class.

    """
    def tree_creation(self, arr, variables, cardinalities):
        return values.Tree.from_array(arr, variables, cardinalities)

    def test_prune(self):
        arr = np.array([[1, 6], [2, 2]])
        tree = values.Tree.from_array(arr, ['0', '1'], {'0': 2, '1': 2})
        tree2 = tree.copy()
        
        self.assertEqual(tree.root.size(), 7)  # Complete node 4 + 2 + 1
        tree.prune()
        self.assertEqual(tree.root.size(), 5)  # prune two leaves
        tree.unprune()
        self.assertEqual(tree, tree2)  

    def test_exceptions_from_array(self):
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([]), [], {})
        with self.assertRaises(ValueError):
            values.Tree.from_array(np.array([1]), ['A'], {'A': 2})

class TableTreeTestCase(TreeTestCase, unittest.TestCase):
    """
    Test Class for values.Tree class.

    """
    def tree_creation(self, arr, variables, cardinalities, *, table_size = 1):
        return values.TableTree.from_array(arr, variables)



    def test_exceptions_from_array(self):
        with self.assertRaises(ValueError):
            values.TableTree.from_array(np.array([]), [])
        with self.assertRaises(ValueError):
            values.TableTree.from_array(np.array([1]), ['A', 'B'])

    def test_prune(self):
        arr = np.array([[1, 1], [1, 1]])
        tree = values.TableTree.from_array(arr, ['0', '1'], table_size = 1)
        tree2 = tree.copy()

        self.assertEqual(tree.root.size(), 3)  # Complete node 4 + 2 + 1
        tree.prune()
        self.assertEqual(tree.root.size(), 1)  # prune two leaves
        tree.unprune()
        self.assertEqual(tree, tree2)  


if __name__ == '__main__':
    unittest.main()
