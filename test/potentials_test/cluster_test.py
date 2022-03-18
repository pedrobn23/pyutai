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
from potentials import cluster, element

class ClusterTestCase(unittest.TestCase):
    """
    Test Class for cluster.Cluster class.
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
            trees.Tree.from_array(arr, variables, self.cardinalities)
            for arr, variables in zip(self.arrays, self.variables)
        ]

        self.clusters = [
            cluster.Cluster.from_tree(tree) for tree in self.trees
        ]

        self.maxDiff = 1000

    def test_from_iterables(self):

        iterable = [
            element.Element(value=0.25, state={
                'A': 0,
                'B': 0
            }),
            element.Element(value=0.25, state={
                'A': 1,
                'B': 0
            }),
            element.Element(value=0.5, state={
                'A': 2,
                'B': 0
            })
        ]
        dictionary = {0.25: {(1, 0), (0, 0)}, 0.5: {(2, 0)}}
        test = cluster.Cluster.from_iterable(iterable,
                                               variables=['A', 'B'],
                                               cardinalities={
                                                   'A': 3,
                                                   'B': 1
                                               })

        # Test equality by double inclusion
        for value in dictionary:
            self.assertTrue(value in test.clusters)

            for element_ in dictionary[value]:
                self.assertTrue(element_ in test.clusters[value])

        for value in test.clusters:
            self.assertTrue(value in dictionary)

            for element_ in test.clusters[value]:
                self.assertTrue(element_ in dictionary[value])

    def test_reduction(self):
        iterable = [
            element.Element(value=0.125, state={
                'A': 1,
                'B': 0
            }),
            element.Element(value=0.125, state={
                'A': 0,
                'B': 0
            }),
            element.Element(value=0.925, state={
                'A': 2,
                'B': 1
            }),
            element.Element(value=0.25, state={
                'A': 2,
                'B': 0
            }),
        ]

        test = cluster.Cluster.from_iterable(iterable,
                                               variables=['A', 'B'],
                                               cardinalities={
                                                   'A': 3,
                                                   'B': 2
                                               })


    def test_array(self):
        nums = itertools.chain(range(10), range(10), range(10))
        array = np.array(list(nums)).reshape(3, 2, 5)

        clt = cluster.Cluster.from_array(array, ['A', 'B', 'C'])

        self.assertTrue(np.array_equal(array, clt.array()))


if __name__ == '__main__':
    unittest.main()
