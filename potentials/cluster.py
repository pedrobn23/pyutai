"""
Cluster module implements k-meas cluster reduction of potentials. WIP at this moment.

[1] Wang, Haizhou & Song, Mingzhou. (2011). Ckmeans.1d.dp: Optimal k-means Clustering
in One Dimension by Dynamic Programming. The R Journal. 3. 29-33. 10.32614/RJ-2011-015.
"""
import collections
import dataclasses
import itertools
import statistics

from typing import Dict, Iterable, List

import numpy as np

from pyutai import distances
from potentials import reductions, element

@dataclasses.dataclass
class Cluster:
    """cluster.Potential defines a cluster-based potential.

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
        """Create a cluster from a iterable object."""
        cluster = collections.defaultdict(set)
        for element in iter_:
            # Transfor the assigment into a tuple for it to be hashable
            state = tuple(element.state[var] for var in variables)

            cluster[element.value].add(state)

        return cls(cluster, variables, cardinalities)

    @staticmethod
    def _iterable_from_array(array: np.ndarray, variables: List[str]):
        """Adapter that creates new iterable from np.ndarray"""
        for position, value in np.ndenumerate(array):
            state = dict(zip(variables, position))
            yield element.Element(value=value, state=state)

    @classmethod
    def from_array(cls, array: np.ndarray, variables=None):
        """Create a cluster from a numpy ndarray"""
        if variables is None:
            variables = [i for i, _ in enumerate(array.shape)]

        cardinalities = dict(zip(variables, array.shape))
        iterable = cls._iterable_from_array(array, variables)

        return cls.from_iterable(iterable, variables, cardinalities)

    @classmethod
    def from_tree(cls, tree):
        """create a cluster from a Tree"""
        return cls.from_iterable(tree, list(tree.variables), tree.cardinalities)

    def _ordered_elements(self):
        cluster_values = sorted(list(self.clusters.keys()))
        elements = [
            element.Element(state, value)
            for value in cluster_values
            for state in self.clusters[value]
        ]
        return elements

    @staticmethod
    def _mean(elements, start, stop):
        return statistics.mean(
            (e.value for e in itertools.islice(elements, start, stop)))

    def reduce_cluster(self, clusters):
        """Create a new potential, with only <cluster> clusters.

        To make that it uses an optimal algorithm that select what is the
        aproximation that better reduce the euclidean distance.
        """
        elements = self._ordered_elements()
        cluster_list = reductions.optimal_cluster(
            elements,
            clusters,
            distance=distances.iterative_euclidean(elements))

        clusters = collections.defaultdict(set)
        for start, stop in cluster_list:
            value = type(self)._mean(elements, start, stop)
            for element in itertools.islice(elements, start, stop):
                clusters[value].add(element.state)

        return type(self)(clusters=clusters,
                          variables=self.variables,
                          cardinalities=self.cardinalities)

    def __iter__(self):
        """Returns an iterator over the values of the Tree.

        Returns:
            element.Element: with the configuration of states variables and the associated value.
        """
        for value, cluster in self.clusters.items():
            for element in cluster:
                indexes = dict(zip(self.variables, element))
                yield element.Element(indexes, value)

    def array(self):
        """Return an np.ndarray with the elements of the cluster."""

        shape = tuple(self.cardinalities[var] for var in self.variables)
        array = np.zeros(shape)

        for value, cluster in self.clusters.items():
            for element in cluster:
                array[element] = value

        return array
