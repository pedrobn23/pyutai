"""
Cluster module implements k-meas cluster reduction of potentials. WIP at this moment.

[1] Wang, Haizhou & Song, Mingzhou. (2011). Ckmeans.1d.dp: Optimal k-means Clustering
in One Dimension by Dynamic Programming. The R Journal. 3. 29-33. 10.32614/RJ-2011-015.
"""
import bisect
import collections
import dataclasses
import math
import itertools
import statistics

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pprint

from pyutai import distances
from potentials import reductions, element


@dataclasses.dataclass(order=True)
class Grain:
    start: Tuple[int]
    end: Tuple[int]

    def __contains__(self, num: Tuple[int]):
        return self.start <= num and num <= self.end

    @classmethod
    def from_tuple(cls, tup):
        return cls(start=tup[0], end=tup[1])


@dataclasses.dataclass
class ValueGrains:
    """
    ValueGrains

    Attributes:
    """

    value_grains: Dict[float, List[Grain]]
    variables: List[str]
    cardinalities: Dict[str, int]

    @staticmethod
    def _max_tuple(indexes: Tuple) -> Tuple:
        if isinstance(indexes, tuple):
            return tuple(math.inf for _ in indexes)
        else:
            return math.inf

    def access(self, indexes: Dict[str, int]) -> float:
        """Retrieve a value from a dictionary mapping."""

        if isinstance(indexes, dict):
            indexes = tuple(indexes[var] for var in self.variables)

        division_grain = Grain(indexes, type(self)._max_tuple(indexes))
        for value, grain_list in self.value_grains.items():
            # division_grain is the smallest grain such that:
            # -  is bigger than any grain that could contain indexes
            #    * Note that math.inf is not a valid value in real grains
            index = bisect.bisect_left(grain_list, division_grain)

            if index > 0:  # if smaller than any grain
                if indexes in grain_list[index - 1]:
                    return value
                

        print(self)
        print('division grain:', division_grain)
        for value, grain_list in self.value_grains.items():
            division_grain = Grain(indexes, type(self)._max_tuple(indexes))
            index = bisect.bisect_left(grain_list, division_grain)

            print('grain list:', grain_list)
            print('index:', index)

        raise ValueError(
            f'Index configuration {zip(self.variables, indexes)} not found.')

    def __iter__(self):
        for value, grains_list in self.value_grains.items():
            for grain in grains_list:
                element_ = grain.start
                end = self.next_element(grain.end)
                while element_ != end:
                    yield element.TupleElement(element_, value)
                    element_ = self.next_element(element_)

    def _iter(self, grain: Grain):
        element_ = grain.start
        end = self.next_element(grain.end)
        while element_ != end:
            yield element_
            element_ = self.next_element(element_)

    def array(self):
        """Return an np.ndarray with the elements of the cluster."""

        shape = tuple(self.cardinalities[var] for var in self.variables)
        array = np.zeros(shape)

        for element_ in self:
            array[element_.state] = element_.value

        return array

    @staticmethod
    def _next_element(tuple_, variables: List, cardinalities: Dict[str, int]):
        if len(tuple_) != len(variables):
            raise ValueError(f'Variable list {variables:r} does not match' +
                             f'provided tuple {tuple_:r}')

        tuple_ = list(tuple_)
        for index, variable in enumerate(variables):
            if tuple_[index] + 1 != cardinalities[variable]:
                tuple_[index] += 1
                return tuple(tuple_)
            else:
                tuple_[index] = 0

        return tuple(tuple_)

    def next_element(self, tuple_):
        return type(self)._next_element(tuple_, self.variables,
                                        self.cardinalities)

    @classmethod
    def _grains_from_sorted_list(cls, sorted_list: List, variables: List,
                                 cardinalities: Dict[str, int]) -> List[Grain]:
        """Generate a sorted grain list from a sorted list"""

        # Special cases for sort lists
        if not sorted_list:
            raise ValueError(f'Excepted non empty list, got {sorted_list}')
        elif len(sorted_list) == 1:
            return [Grain(start=sorted_list[0], end=sorted_list[0])]
        elif len(sorted_list) == 2:
            return [Grain(start=sorted_list[0], end=sorted_list[1])]

        # General case
        grain_list = []
        grain = Grain(start=sorted_list[0], end=sorted_list[1])

        # for the second element onward
        for element in itertools.islice(sorted_list, 2, None):
            if element == cls._next_element(grain.end, variables,
                                            cardinalities):
                grain.end = element
            else:
                grain_list.append(grain)
                grain = Grain(element, element)

        grain_list.append(grain)
        return grain_list

    @classmethod
    def from_iterable(cls, iter_: Iterable[element.Element], variables,
                      cardinalities):
        """Create a cluster from a iterable object."""

        # Group element by values
        cluster = collections.defaultdict(list)
        for element in iter_:
            if isinstance(element.state, dict):
                state = tuple(element.state[var] for var in variables)
            else:
                state = element.state
            cluster[element.value].append(state)

        # Transform lists into grain lists
        value_grains = collections.defaultdict(list)
        for value in cluster:
            cluster[value].sort()
            value_grains[value] = cls._grains_from_sorted_list(
                cluster[value], variables, cardinalities)

        return cls(value_grains, variables, cardinalities)

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

    def __str__(self):
        ret = 'Value Grain object:\n'
        ret += f'  - Variables: {self.variables}\n'
        ret += f'  - Cardinalities: {self.cardinalities}\n'
        ret += f'  - Grains: \n'

        for value, grains in self.value_grains.items():
            ret += f'       * value {value}:\n'
            ret += f'       '
            ret += pprint.pformat(grains, indent=8)
            ret += '\n'
        return ret
