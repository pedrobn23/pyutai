"""
Cluster module implements k-meas cluster reduction of potentials. WIP at this moment.

[1] Wang, Haizhou & Song, Mingzhou. (2011). Ckmeans.1d.dp: Optimal k-means Clustering
in One Dimension by Dynamic Programming. The R Journal. 3. 29-33. 10.32614/RJ-2011-015.
"""
import collections
import dataclasses
import itertools
import statistics

from typing import Dict, Iterable, List, Tuple

import numpy as np

from pyutai import distances
from potentials import reductions, element, utils
from experiments import networks


def _mean(elements: List[element.TupleElement], start: int, stop: int):
    """return the mean value of the values from start to stop slice"""
    return statistics.mean(
        (e.value for e in itertools.islice(elements, start, stop)))


@dataclasses.dataclass
class Cluster:
    """Cluster defines a cluster-based potential.

    Attributes:
        clusters: clusters is a dictionary from the value to the set of
           variable-configurations that has such value.
        variables: list of variables to be considered.
        cardinalities: map from variable name to associated cardinality.
            Same cardinality map may be used for different potentials.
    """

    clusters: Dict[float, set]
    variables: List[str]
    cardinalities: Dict[str, int]

    def access(self, indexes: Dict[str, int]) -> float:
        """Retrieve a value from a dictionary mapping."""
        if isinstance(indexes, dict):
            indexes = tuple(indexes[var] for var in self.variables)

        for value, index_set in self.clusters.items():
            if indexes in index_set:
                return value

        raise ValueError('Index configuration not found.')

    @classmethod
    def from_iterable(cls, iter_: Iterable[element.Element], variables,
                      cardinalities):
        """Create a cluster from a iterable object of dict."""
        cluster = collections.defaultdict(set)
        for element in iter_:
            # Transfor the assigment into a tuple for it to be hashable
            if isinstance(element.state, dict):
                state = tuple(element.state[var] for var in variables)
            else:
                state = element.state

            cluster[element.value].add(state)

        return cls(cluster, variables, cardinalities)

    @staticmethod
    def _iterable_from_array(array: np.ndarray, variables: List[str]):
        """Adapter that creates new iterable from np.ndarray"""
        for position, value in np.ndenumerate(array):
            yield element.Element(value=value, state=position)

    @classmethod
    def from_array(cls, array: np.ndarray, variables=None):
        """Create a cluster from a numpy ndarray"""
        if variables is None:
            variables = [i for i, _ in enumerate(array.shape)]

        cardinalities = dict(zip(variables, array.shape))
        iterable = cls._iterable_from_array(array, variables)

        return cls.from_iterable(iterable, variables, cardinalities)

    def _ordered_elements(self):
        cluster_values = sorted(list(self.clusters.keys()))
        elements = [
            element.Element(state, value) for value in cluster_values
            for state in self.clusters[value]
        ]
        return elements

    def _new_clusters(elements: List[element.TupleElement],
                      cluster_list: List[Tuple[int, int]]):
        clusters = collections.defaultdict(set)
        for start, stop in cluster_list:
            value = _mean(elements, start, stop)
            for element in itertools.islice(elements, start, stop):
                clusters[value].add(element.state)

    def reduce(self, n_clusters: int):
        """Create a new potential, with only <cluster> clusters.

        To make that it uses an optimal algorithm that select what is the
        aproximation that better reduce the euclidean distance.
        """
        elements = self._ordered_elements()
        reduction = reductions.Reduction.from_elements(elements,
                                                       threshold=len(
                                                           self.clusters))

        cluster_list = reductions.optimal_partition(n_clusters)
        clusters = self._new_clusters(elements, cluster_list)

        return type(self)(clusters=new_clusters,
                          variables=self.variables,
                          cardinalities=self.cardinalities)

    def __iter__(self):
        """Returns an iterator over the values of the Tree.

        Yields:
            element.TupleElement: with the configuration of states variables and the associated value.
        """
        for value, cluster in self.clusters.items():
            yield from cluster

    def iter(self, *, dict_elements = False):
        if dict_elements:
            for element_ in self:
                indexes = dict(zip(self.variables, element_))
                yield element.Element(indexes, element_.value)
        else:
            yield from self

        
                
    def array(self):
        """Return an np.ndarray with the elements of the cluster."""

        shape = tuple(self.cardinalities[var] for var in self.variables)
        array = np.zeros(shape)

        for value, cluster in self.clusters.items():
            for element in cluster:
                array[element] = value

        return array


if __name__ == '__main__':
    #    Programar experimento y dejarlo en casa
    for cpd in networks.medical():
        clt = cluster.Cluster.from_array(cpd.values, cpd.variables)

        original_size = utils.total_size(clt)
        # implementar error based cluster reduction
