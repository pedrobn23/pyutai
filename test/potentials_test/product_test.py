"""Module that implements automatic test cases for functions in cluster.py and
reduction.py
"""

import copy
import itertools
import math
import numpy as np
import statistics
import unittest

from pyutai import trees, nodes, distances
from potentials import element, cluster, valuegrains, indexmap, indexpairs, operations


class ProductTestCase:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_product(self):
        card = {'A': 2, 'B': 2, 'C': 2}

        variables1 = ['A', 'B']
        variables2 = ['B', 'C']
        variables3 = ['A', 'B', 'C']

        arr1 = np.array([[1, 2], [2, 3]])
        arr2 = np.array([[4, 5], [0, 0]])
        arr3 = np.array([[[4, 5], [0, 0]], [[8, 10], [0, 0]]])

        obj1 = self.cls.from_array(arr1, variables1)
        obj2 = self.cls.from_array(arr2, variables2)

        obj3 = self.cls.from_array(arr3, variables3)

        obj4 = self.cls.from_iterable(*operations.combine(obj1, obj2))

        for state in itertools.product(
                *[range(obj3.cardinalities[var]) for var in obj3.variables]):
            # using dict to avoid variables missorder
            state = dict(zip(obj3.variables, state))
            x3 = obj3.access(state)
            x4 = obj4.access(state)

            self.assertEqual(x3, x4)


class ProductClusterTestCase(unittest.TestCase, ProductTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = cluster.Cluster
        self.maxDiff = 1000


class ProductValueGrainsTestCase(unittest.TestCase, ProductTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = valuegrains.ValueGrains
        self.maxDiff = 1000


class ProductIndexMapTestCase(unittest.TestCase, ProductTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = indexmap.IndexMap
        self.maxDiff = 1000


class ProductIndexPairsTestCase(unittest.TestCase, ProductTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = indexpairs.IndexPairs
        self.maxDiff = 1000


if __name__ == '__main__':
    unittest.main()
