"""Selector helper for Tree construction

This module contains some functions that ease out custom var selection.


TODO: Mutual information selector.
"""

from typing import List, Callable
from scipy import stats

import numpy as np

from pyutai import trees


def _normalize(data: np.ndarray):
    return data / data.sum()


def _filter(data: np.ndarray, selections: trees.IndexSelection,
            variables: List[str]):
    filter_ = tuple(
        slice(None) if var not in selections else selections[var]
        for var in variables)

    variables_ = [var for var in variables if var not in selections]

    return data[filter_], variables_


def _restriction_iterator(data: np.ndarray, variable: int):
    cardinality = data.shape[variable]
    for state in range(cardinality):
        filter_ = tuple(
            slice(None) if var != variable else state
            for var, _ in enumerate(data.shape))
        yield data[filter_]


def minimal_selector(data: np.ndarray,
                     variables: List[str],
                     _evaluator: Callable,
                     normalize: bool = False) -> trees.VarSelector:
    """Generates a VarSelector that minimizes _evaluator score.."""

    if normalize:
        data = _normalize(data)

    def variable_selector(previous_selections: trees.IndexSelection = None):
        if previous_selections is None:
            previous_selections = {}

        filtered_data, filtered_variables = _filter(data, previous_selections,
                                                    variables)

        results = [
            _evaluator(filtered_data, variable)
            for variable, _ in enumerate(filtered_variables)
        ]
        return filtered_variables[np.argmin(results)]

    return variable_selector


def variance(data: np.ndarray, variables: List[str]):
    """Generates a VarSelector based on the minimum entropy principle."""

    def variance_(data, variable):
        return sum(
            restricted_data.var()**2
            for restricted_data in _restriction_iterator(data, variable))

    return minimal_selector(data, variables, variance_, normalize=False)


def entropy(data: np.ndarray, variables: List[str]):
    """Return a new VarSelector based on the minimun entropy principle."""

    def entropy_(data, variable):
        return sum(
            stats.entropy(data.flatten())
            for restricted_data in _restriction_iterator(data, variable))

    return minimal_selector(data, variables, entropy_, normalize=True)
