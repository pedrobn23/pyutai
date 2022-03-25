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


class MarginalizeTestCase:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def test_marg(self):
        card = {'A': 2, 'B': 2}

        arrs = [
            np.array([[1, 2], [2, 3]]),
            np.array([[4, 5], [0, 0]]),
            np.array([[1, 1], [1, 1]])
        ]
        marg_arrs = [arr.sum(axis=1) for arr in arrs]

        for arr, marg_arr in zip(arrs, marg_arrs):
            obj = self.cls.from_array(arr, ['A','B'])
            marg_obj = self.cls.from_array(marg_arr, ['A'])
            obj2 = self.cls.from_iterable(*operations.marginalize(obj, 'B'))

            for state in itertools.product(*[range(obj2.cardinalities[var]) for var in obj2.variables]):
                # using dict to avoid variables missorder
                state = dict(zip(obj2.variables, state))
                x3 = obj2.access(state)
                x4 = marg_obj.access(state)
                
                self.assertEqual(x3,x4)

            
class MarginalizeClusterTestCase(unittest.TestCase, MarginalizeTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = cluster.Cluster
        self.maxDiff = 1000


class MarginalizeValueGrainsTestCase(unittest.TestCase, MarginalizeTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = valuegrains.ValueGrains
        self.maxDiff = 1000


class MarginalizeIndexMapTestCase(unittest.TestCase, MarginalizeTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = indexmap.IndexMap
        self.maxDiff = 1000


class MarginalizeIndexPairsTestCase(unittest.TestCase, MarginalizeTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = indexpairs.IndexPairs
        self.maxDiff = 1000
        
if __name__ == '__main__':
    unittest.main()
