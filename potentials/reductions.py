"""
Reduction module implements k-meas cluster reduction of potentials. WIP at this moment.

[1] Wang, Haizhou & Song, Mingzhou. (2011). Ckmeans.1d.dp: Optimal k-means Clustering
    in One Dimension by Dynamic Programming. The R Journal. 3. 29-33. 10.32614/RJ-2011-015.
[2] PyUTAI group. (2022)  https://leo.ugr.es/emlpa/meetings/granada-24-25-1-22/pbonilla.pdf
"""

import math

from typing import Callable, List

import tqdm
import numpy as np

from pyutai import distances
from potentials import cluster, element

def _select_best_option(element, cluster, errors, distance):
    """Select the best options to distribute elements into cluster"""
    min_error = math.inf
    min_index = math.inf

    for last_element in range(cluster - 1, element):
        previous_clustering_error = errors[last_element][cluster - 1]
        new_cluster_error = distance(last_element, element)

        potential_error = previous_clustering_error + new_cluster_error

        if min_error >= potential_error:
            min_error = potential_error
            min_index = last_element

    return min_error, min_index


def _cluster_list(indexes):
    """Given a table of indexes, generates cluster sequence."""
    elements_limit, clusters_limit = indexes.shape

    clusters = []
    clusters_left = clusters_limit - 1
    start = 0
    end = int(elements_limit - 1)

    for _ in range(clusters_limit - 1):
        start = int(indexes[end, clusters_left])

        clusters.append((start, end))

        clusters_left -= 1
        end = start

    return clusters


def optimal_cluster(elements: List[element.Element],
                    clusters: int,
                    *,
                    distance: Callable = None,
                    interactive: bool = False):
    """Generates a list of indices denoting optimal clustering.

    Receives a list of elements ordered by value, and return the indexes
    of the optimal clustering.


    Example:
    TODO
    """
    if distance is None:
        distance = distances.iterative_euclidean(elements)

    n_elements = len(elements)
    if clusters > n_elements:
        raise ValueError(f'Provided list has not enough elements {n_elements}' +
                         f'to create {clusters} clusters')

    errors = np.zeros(shape=(n_elements + 1, clusters + 1))
    indexes = np.zeros(shape=(n_elements + 1, clusters + 1))

    # Nice terminal update if interactive
    if interactive:
        print('Progress of reduction:')
        elements_indexes = tqdm.tqdm(range(len(elements) + 1))
    else:
        elements_indexes = range(len(elements) + 1)

    # Dynamic Programing step
    for element in elements_indexes:
        for cluster in range(clusters + 1):

            # If there is nothing to do
            if element == 0 or cluster == 0:
                errors[element, cluster] = 0
                indexes[element, cluster] = -1

            # One element in each cluster
            elif cluster == element:
                errors[element][cluster] = 0
                indexes[element][cluster] = element - 1

            # Empty clusters
            elif cluster > element:
                pass

            # All element in one cluster
            elif cluster == 1:
                errors[element][cluster] = distance(0, element)
                indexes[element][cluster] = 0

            # General case
            else:
                error, index = _select_best_option(element, cluster, errors,
                                                   distance)
                errors[element][cluster] = error
                indexes[element][cluster] = index

    # final formating
    return _cluster_list(indexes)
