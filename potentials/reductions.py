"""
Reduction module implements k-meas cluster reduction of potentials. WIP at this moment.

[1] Wang, Haizhou & Song, Mingzhou. (2011). Ckmeans.1d.dp: Optimal k-means Clustering
    in One Dimension by Dynamic Programming. The R Journal. 3. 29-33. 10.32614/RJ-2011-015.
[2] PyUTAI group. (2022)  https://leo.ugr.es/emlpa/meetings/granada-24-25-1-22/pbonilla.pdf
"""

import math
import dataclasses
import itertools
import statistics
import warnings

from typing import Callable, List, Tuple

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


@dataclasses.dataclass
class Reduction:
    """This class provide different information about the reduction process.

    It contains an error 
    """
    errors: List[List[float]]
    indexes: List[List[int]]
    elements: List[element.TupleElement]
    # Element list space [O(n)] is dominated matrix space [O(n**2)].


    @property
    def precomputed_partitions(self):
        """Amount of precomputed partitions on initialization."""
        return len(self.indexes[0,slice(None)])
    
    @classmethod
    def from_elements(cls,
                      elements: List[element.TupleElement],
                      *,
                      threshold: int = None,
                      distance: Callable = None,
                      interactive: bool = False):
        """Initialize a matrix with diferent of indices denoting optimal clustering.
        
        Uses a variation of the dynamic programming method proposed in [1] to generate
        a matrix of minimal errors of partitioning 

        Due to otherwise heavy-processing if miss-used, threshold is set to 
        math.sqrt(len(elements)). For most cases, it is recommend to set is to the 
        original amount of different values.
        

        Example:
        TODO
        """
        if distance is None:
            distance = distances.iterative_kullback(elements)

        if threshold is None:
            threshold = math.isqrt(len(elements))

        n_elements = len(elements)

        if threshold > n_elements:
            raise ValueError(
                f'Provided list has not enough elements {n_elements}' +
                f'to create {threshold} clusters')
        n_clusters = threshold

        errors = np.zeros(shape=(n_elements + 1, n_clusters + 1))
        indexes = np.zeros(shape=(n_elements + 1, n_clusters + 1))

        # Nice terminal update if interactive
        if interactive:
            print('Progress of reduction:')
            elements_indexes = tqdm.tqdm(range(len(elements) + 1))
        else:
            elements_indexes = range(len(elements) + 1)

        # Dynamic Programing step
        for element in elements_indexes:
            for cluster in range(n_clusters + 1):

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
                    error, index = _select_best_option(element, cluster,
                                                       errors, distance)
                    errors[element][cluster] = error
                    indexes[element][cluster] = index

        return cls(errors=errors, indexes=indexes, elements=list(elements))

    
    def error(self, n_cluster: int):
        """Returns the minimum error of partitioning the set of elements in <n_cluster>."""
        return self.errors[len(self.elements)][n_cluster]

    def reduction(self, error: float = 0.05) -> int:
        """Minimal partitions that creates an aproximation with, at most, 
        <error> error. Error is calculated with the distance function provided to 
        Reduction.from_elements(). If no distance was provided, iterative kullback
        is used."""

        for n_cluster in range(1, self.precomputed_partitions+1):
            if self.error(n_cluster) < error:
                return self.optimal_partition(n_cluster)

        raise ValueError(f'No partition has been found for error {error}.' +
                         ' Consider increasing threshold value in' +
                         ' Reduction.from_elements() method.')

    def optimal_partition(self, clusters: int):
        if clusters < 0:
            raise ValueError(
                f'Number of cluster should be non-negative integer, got : {clusters}.'
            )
        if clusters > (max_clusters := len(self.indexes)):
            raise ValueError(
                f'Precomputation done for cluster value up to {max_clusters}.')

        cluster_list = []
        clusters_left = clusters

        start = 0
        end = len(self.elements)
        for _ in range(clusters - 1):
            start = int(self.indexes[end, clusters_left])

            cluster_list.append((start, end))

            clusters_left -= 1
            end = start

        cluster_list.append((0, end))
        return cluster_list

    def _mean(self, start: int, stop: int):
        """return the mean value of the values from start to stop slice"""
        return statistics.mean(
            (e.value for e in itertools.islice(self.elements, start, stop)))

    def _TupleElements(self, *, vars_):
        if isinstance(self.elements[0].state, tuple):
            return self.elements

        elif isinstance(self.elements[0].state, dict):
            if vars_ is None:
                warnings.warn('Infering order of provided variables. To change' +
                              'this behaviour provide vars_ argument.')
                vars_ = sorted(list(self.elements[0].state.keys()))

            def dict_to_tuple(dictionary : dict) -> tuple:
                return tuple(dictionary[key] for key in vars_)

            return [element.TupleElement(state=dict_to_tuple(element_.state),
                                         value=element_.value)
                    for element_ in self.elements]

        else:
            raise AttributeError('Elements should have dictionary or tuple states, got:{self.elements[0].state}')
    
    def array(self, n_cluster: int, shape: Tuple[int], *, vars_ : List[str] = None):
        """Return np.ndarray with optimal partition carried out.
        
        Creates a new array with the new value for elements in partition.
        If an element is not provided by original cluster it is suppossed
        to have a 0 value a thus should remain."""

        try:
            cluster_list = self.optimal_partition(n_cluster)
        except ValueError as value_error:
            raise ValueError(
                'Partition could not be carried out') from value_error

        array = np.zeros(shape)

        for start, stop in cluster_list:
            value = self._mean(start, stop)
            for element_ in itertools.islice(self._TupleElements(vars_=vars_), start, stop):
                try:
                    array[element_.state] = value
                except IndexError as index_error:
                    print(element_)
                    raise ValueError(
                        'Provided shape {shape} does not fit element {element}'
                    ) from index_error

        return array
