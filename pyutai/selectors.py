"""Selector helper for Tree construction

This module contains some functions that ease out custom var selection.


TODO: Mutual information selector.
"""

from typing import List, Callable
from scipy import stats

import numpy as np

from pyutai import values


def _normalize(data: np.ndarray):
    return data / data.sum()


def _filter(data: np.ndarray, selections: values.IndexSelection,
            variables: List[str]):
    filter_ = tuple(
        slice(None) if var not in selections else selections[var]
        for var in variables)

    variables_ = [var for var in variables if var not in selections]

    return data[filter_], variables_


def _restriction_iterator(data: np.ndarray, variable: int):
    cardinality = data.shape[variable]
    for state in range(cardinality):
        filter_ = (slice(None) for _ in data.shape)
        filter_[variable] = state
        yield data[filter_]


def minimal_selector(data: np.ndarray,
                     variables: List[str],
                     _evaluator: Callable,
                     normalize: bool = False) -> values.VarSelector:
    """Generates a VarSelector that minimizes _evaluator score.."""

    if normalize:
        data = _normalize(data)

    def variable_selector(previous_selections: values.IndexSelection = None):
        if previous_selections is None:
            previous_selections = {}

        filtered_data, filtered_variables = _filter(data, previous_selections,
                                                    variables)

        results = []
        for variable in filtered_variables:
            error = sum(
                _evaluator(data_)
                for data_ in _restriction_iterator(filtered_data, variable))
            results.append(error)

        return filtered_variables[np.argmin(results)]

    return variable_selector


def variance(data: np.ndarray, variables: List[str]):
    """Generates a VarSelector based on the minimum entropy principle."""

    _variance = lambda data: data.var()
    return minimal_selector(data, variables, _variance, normalize=False)


def entropy(data: np.ndarray, variables: List[str]):
    """Return a new VarSelector based on the minimun entropy principle."""

    _entropy = lambda data: stats.entropy(data.flatten())
    return minimal_selector(data, variables, _entropy, normalize=True)
