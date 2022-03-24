"""Module that implements automatic test cases for functions in cluster.py and
reduction.py
"""

import copy
import itertools
import math
import numpy as np
import statistics
import unittest

from typing import List

from pyutai import trees, nodes, distances
from potentials import cluster, element, reductions


def _kullback(elements: List[element.TupleElement]):
    """Helper to check Kullback-Leibler distance."""
    elements = [element_.value for element_ in elements]
    mean = statistics.mean(elements)
    return sum(element * (np.log(element) - np.log(mean))
               for element in elements if element != 0)


class ReductionImplementationCase(unittest.TestCase):
    """
    Test Class for cluster.Cluster class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.elements = [
            element.Element(value=0.25, state={
                'A': 0,
                'B': 0
            }),
            element.Element(value=0.125, state={
                'A': 0,
                'B': 1
            }),
            element.Element(value=0.25, state={
                'A': 0,
                'B': 2
            }),
            element.Element(value=0.125, state={
                'A': 1,
                'B': 0
            }),
            element.Element(value=0.25, state={
                'A': 1,
                'B': 1
            }),
            element.Element(value=0.0625, state={
                'A': 1,
                'B': 2
            }),
        ]

        self.elements = sorted(self.elements, key=lambda x: x.value)
        self.reduction = reductions.Reduction.from_elements(self.elements,
                                                            threshold=3)

    def test_optimal_partition(self):

        self.assertEqual(sorted(self.reduction.optimal_partition(1)), [(0, 6)])
        self.assertEqual(sorted(self.reduction.optimal_partition(2)), [(0, 3),
                                                                       (3, 6)])
        self.assertEqual(sorted(self.reduction.optimal_partition(3)), [(0, 1),
                                                                       (1, 3),
                                                                       (3, 6)])

    def test_error(self):
        np.testing.assert_almost_equal(self.reduction.error(0), 0)
        np.testing.assert_almost_equal(self.reduction.error(1),
                                       _kullback(self.elements))
        np.testing.assert_almost_equal(
            self.reduction.error(2),
            _kullback(self.elements[0:3]) + _kullback(self.elements[3:6]))
        np.testing.assert_almost_equal(
            self.reduction.error(3),
            _kullback(self.elements[0:1]) + _kullback(self.elements[1:3]) +
            _kullback(self.elements[3:6]))

    def test_reduction(self):
        self.assertEqual(sorted(self.reduction.reduction(0.05)), [(0, 3),
                                                                  (3, 6)])
        self.assertEqual(sorted(self.reduction.reduction(0.1)), [(0, 3),
                                                                 (3, 6)])
        self.assertEqual(sorted(self.reduction.reduction(0.5)), [(0, 6)])
        self.assertEqual(sorted(self.reduction.reduction(1)), [(0, 6)])

    def test_array(self):
        arr1 = self.reduction.array(1, (2, 3), vars_=['A', 'B'])
        arr1_check = np.array([[0.17708333, 0.17708333, 0.17708333],
                               [0.17708333, 0.17708333, 0.17708333]])
        np.testing.assert_almost_equal(arr1, arr1_check)

        arr1 = self.reduction.array(2, (2, 3), vars_=['A', 'B'])
        arr1_check = np.array([[0.25, 0.1041667, 0.25],
                               [0.1041667, 0.25, 0.1041667]])
        np.testing.assert_almost_equal(arr1, arr1_check)

        arr1 = self.reduction.array(3, (2, 3), vars_=['A', 'B'])
        arr1_check = np.array([[0.25, 0.125, 0.25], [0.125, 0.25, 0.0625]])
        np.testing.assert_almost_equal(arr1, arr1_check)


if __name__ == '__main__':
    unittest.main()
