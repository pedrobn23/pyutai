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
from potentials import reductions, element


@dataclasses.dataclass
class IndexMap:
    """
    ValueGrains

    Attributes:
    """

    values: Dict[Tuple[int], float]
    variables: List[str]
    cardinalities: Dict[str, int]

    def access(self, indexes: Dict[str, int]) -> float:
        """Retrieve a value from a dictionary mapping."""
        if isinstance(indexes, dict):
            indexes = tuple(indexes[var] for var in self.variables)

        try:
            return self.values[indexes]
        except KeyError as ke:
            raise ValueError(
                f'Index configuration {zip(self.variables, indexes)} not found.'
            )

    def __iter__(self):
        for state, value in self.values.items():
            yield element.TupleElement(state, value)

    def array(self):
        """Return an np.ndarray with the elements of the cluster."""

        shape = tuple(self.cardinalities[var] for var in self.variables)
        array = np.zeros(shape)

        for element_ in self:
            array[element_.state] = element_.value

        return array

    @classmethod
    def from_iterable(cls, iter_: Iterable[element.Element], variables,
                      cardinalities):
        """Create a cluster from a iterable object."""

        values = {}
        for element in iter_:
            if isinstance(element.state, dict):
                state = tuple(element.state[var] for var in variables)
            else:
                state = element.state

            values[state] = element.value
        return cls(values, variables, cardinalities)

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
