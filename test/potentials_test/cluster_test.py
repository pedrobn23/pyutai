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


class ClusterImplementationCase(unittest.TestCase):
    """
    Test Class for cluster.Cluster class.
    """

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


if __name__ == '__main__':
    unittest.main()
