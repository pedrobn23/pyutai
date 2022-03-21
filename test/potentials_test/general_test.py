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
from potentials import element, cluster, valuegrains, indexmap, indexpairs




class GeneralTestCase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def test_access(self):
        nums = itertools.chain(range(10), range(10), range(10))
        array = np.array(list(nums)).reshape(3, 2, 5)

        clt = self.cls.from_array(array, ['A', 'B', 'C'])
        
        for position, value in np.ndenumerate(array):
            self.assertTrue(abs(clt.access(position)-value) < 10**3)

    def test_array(self):
        nums = itertools.chain(range(10), range(10), range(10))
        array = np.array(list(nums)).reshape(3, 2, 5)

        clt = self.cls.from_array(array, ['A', 'B', 'C'])
        self.assertTrue(np.array_equal(array, clt.array()))


class GeneralClusterTestCase(unittest.TestCase, GeneralTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = cluster.Cluster
        self.maxDiff = 1000

class GeneralValueGrainsTestCase(unittest.TestCase, GeneralTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = valuegrains.ValueGrains
        self.maxDiff = 1000


class GeneralIndexMapTestCase(unittest.TestCase, GeneralTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = indexmap.IndexMap
        self.maxDiff = 1000


class GeneralIndexPairsTestCase(unittest.TestCase, GeneralTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = indexpairs.IndexPairs
        self.maxDiff = 1000
