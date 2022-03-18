"""
Module that implements automatic test cases for selectors functions.

TODO: estos test aun no hace nada, pls implementa una opción.
"""
import itertools
import unittest
import numpy as np

from pyutai import values, nodes, selectors


# class EntropySelectorTestCase(unittest.TestCase):
#     """
#     Test Class for Node classes in values.py.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def test_entropy(self):
#         array = np.array([
#             [[2, 2], [1, 1]],
#             [[2, 2], [1, 1]],
#         ])
#         variables = ['A', 'B', 'C']
#         var_selector = selectors.entropy(array, variables)


# class VarianceSelectorTestCase(unittest.TestCase):
#     """
#     Test Class for Node classes in values.py.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def test_variance(self):
#         array = np.array([[[2, 7], [8, 1]], [[3, 2], [1, -1]]])
#         variables = ['A', 'B', 'C']
#         var_selector = selectors.variance(array, variables)



if __name__ == '__main__':
    unittest.main()
