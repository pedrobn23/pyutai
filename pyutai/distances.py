"""
Distances module provide implementations of iterative distance to be used in clustering.

For more information see formulas in [p.11, 1].

[1] PyUTAI group. (2022)  https://leo.ugr.es/emlpa/meetings/granada-24-25-1-22/pbonilla.pdf
[2] https://en.wikipedia.org/wiki/Closure_(computer_programming)
"""

import math
import itertools

import numpy as np

from pyutai import trees


def _euclidean_step():
    """Returns a closure that progresively compute the error of the cluster.


    Returns a closure[2] that receives trees.Element object, and compute
    the quadratic error to the mean. Each call runs in O(1). Step formula
    is based on [1, p.11]

    Example:

    >>> from pyutai import distances, values

    # cluster_error is the closure that compute the error
    >>> cluster_error = distances._euclidean_step()

    # Error of cluster {0}
    >>> cluster_error(trees.Element({'A':0},1))
    0.0

    # Error of cluster {0,1}
    >>> cluster_error(trees.Element({'A':1},2))
    0.5
    """

    error = 0
    median = 0

    element_count = 0
    total_weight = 0

    def function(element: trees.Element):
        nonlocal error, median, element_count, total_weight

        element_count += 1

        delta = (element_count - 1) / element_count
        error += delta * (element.value - median)**2

        median = (element.value + total_weight) / element_count
        total_weight += element.value

        return error

    return function


def _kullback_step():
    """Returns a closure that progresively compute the error of the cluster.

    Returns a closure[2] that receives trees.Element object, and compute
    the quadratic error to the mean. Each call runs in O(1). Step formula
    is based on [1, p.11]

    Example:

    >>> from pyutai import distances, values

    # cluster_error is the closure that compute the error
    >>> cluster_error = distances._kullback_step()

    # Error of cluster {1}
    >>> cluster_error(trees.Element({'A':0},1))
    0.0

    # Error of cluster {0, 1}
    >>> cluster_error(trees.Element({'A':1},0))
    0.6931471805599453
    """

    error = 0
    median = 0

    element_count = 0
    total_weight = 0

    def function(element: trees.Element):
        nonlocal error, median, element_count, total_weight

        if element.value < 0:
            raise ValueError('Kullback Leibler divergence does not accepts ' +
                             f'negative values. Received: {element.value}.')

        element_count += 1
        new_total_weight = total_weight + element.value
        new_median = (element.value + total_weight) / element_count

        if median == 0 and new_median == 0:
            error = 0

        elif median == 0 and new_median != 0:
            # This case can happen only if element.value != 0
            error = element.value * math.log(element.value / new_median)

        elif median != 0 and element.value == 0:
            error += total_weight * math.log(median / new_median)

        else:
            error += total_weight * math.log(
                median / new_median) + element.value * math.log(
                    element.value / new_median)

        total_weight = new_total_weight
        median = new_median

        return error

    return function


def _iterative(elements, _step):
    """Helper that generates the closure

    As it uses steps functions, it only takes O(n**2) time, with n the number of elements,
    to generate the closure
    """

    distances = np.zeros((len(elements), len(elements)))

    for start, _ in enumerate(elements):
        cluster_error = _step()
        for stop, element in itertools.islice(enumerate(elements), start,
                                              len(elements)):
            distances[(start, stop)] = cluster_error(element)

    def closure(start: int, stop: int):
        """Return the error of a cluster from elements start to stop.

        closure(i,j) is equal to distance(elemenst[i,j], [mean]*(j-1)).
        Distance depends on the step function used. The algorithm
        return a memoized distance.
        """
        return distances[start, stop - 1]

    return closure


def iterative_euclidean(elements):
    """Return the error of a cluster from elements start to stop.

    closure(i,j) is equal to sum((elem - mean)**2 for elem in elements).
    Distance depends on the step function used. The algorithm
    return a memoized distance.
    """
    return _iterative(elements, _euclidean_step)


def iterative_kullback(elements):
    """Return the error of a cluster from elements start to stop.

    closure(i,j) is equal to sum(D_KL(elem, mean) for elem in elements).
    Distance depends on the step function used. The algorithm
    return a memoized distance.
    """
    return _iterative(elements, _kullback_step)
