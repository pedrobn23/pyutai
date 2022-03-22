"""Module that implements automatic test cases for distances module
"""

import copy
import itertools
import math
import numpy as np
import statistics
import unittest

from pyutai import trees, nodes, distances


class EUDistanceTestCase(unittest.TestCase):
    """
    Test Class for distances.iterative_euclidean()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [[1, 2, 3, 4, 97, 23, 27, 4, 12], [0, 0, 0, 0, 1]]
        self.elements_list = [[
            trees.Element(state=i, value=val) for i, val in enumerate(arr)
        ] for arr in self.arrays]

    def test_eculidean_step(self):
        for arr, elements in zip(self.arrays, self.elements_list):
            distance = distances._euclidean_step()
            for index, element in enumerate(elements):
                error = distance(element)
                if index != 0:
                    error2 = statistics.variance(arr[0:index + 1]) * (index)
                    self.assertAlmostEqual(error, error2, places=10)

    def test_iterative_euclidean(self):
        for elements, arr in zip(self.elements_list, self.arrays):
            distance = distances.iterative_euclidean(elements)

            for i, _ in enumerate(elements):
                for j in range(i + 1, len(elements)):

                    error1 = distance(i, j + 1)
                    error2 = statistics.variance(arr[i:j + 1]) * (j - i)
                    self.assertAlmostEqual(error1, error2, places=10)


def _kullback(elements):
    """Helper to check Kullback-Leibler distance."""
    mean = statistics.mean(elements)
    return sum(element * (math.log(element) - math.log(mean))
               for element in elements if element != 0)


class KLDistanceTestCase(unittest.TestCase):
    """
    Test Class for distances.iterative_kullback()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = [[1, 2, 3, 4, 97, 23, 27, 4, 12], [0, 0, 0, 0, 1]]
        self.elements_list = [[
            trees.Element(state=i, value=val) for i, val in enumerate(arr)
        ] for arr in self.arrays]

    def test_kullback_step(self):
        for arr, elements in zip(self.arrays, self.elements_list):
            distance = distances._kullback_step()
            for index, element in enumerate(elements):
                error = distance(element)
                if index != 0:
                    error2 = _kullback(arr[0:index + 1])
                    self.assertAlmostEqual(error, error2, places=10)

    def test_iterative_kullback(self):
        for elements, arr in zip(self.elements_list, self.arrays):
            distance = distances.iterative_kullback(elements)
            for i, _ in enumerate(elements):
                for j in range(i, len(elements)):
                    error1 = distance(i, j + 1)
                    error2 = _kullback(arr[i:j + 1])
                    self.assertAlmostEqual(error1, error2, places=10)


if __name__ == '__main__':
    unittest.main()
