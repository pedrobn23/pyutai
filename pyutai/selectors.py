"""Selector helper for Tree construction

This module contains some functions that ease out custom var selection.
"""



from typing import List, Callable
from scipy import stats

import numpy as np

from pyutai import values

def _normalize(data: np.ndarray):
    return data / data.sum()


def minimal_selector(data: np.ndarray,
                     variables: List[str],
                     _evaluator: Callable,
                     normalize: bool = False) -> values.VarSelector:
    """Generates a VarSelector that minimizes _evaluator score.."""

    if normalize:
        data = _normalize(data)

    def var_selector(selections: values.IndexSelection = None):
        if selections is None:
            selections = {}

        var_filter = tuple(
            slice(None) if var not in selections else selections[var]
            for var in variables)

        restricted_data = data[var_filter]

        restricted_variables = [
            var for var in variables if not (var in selections)
        ]

        variances = [
            _evaluator(restricted_data, index)
            for index, _ in enumerate(restricted_variables)
        ]
        minimal_variance = np.argmax(variances)

        return restricted_variables[minimal_variance]

    return var_selector


def _variance(data: np.ndarray, variable: str):

    axes = tuple(x for x in range(data.ndim) if x != variable)
    return data.sum(axis=axes).var()


def variance(data: np.ndarray, variables: List[str]):
    """Generates a VarSelector based on the minimum entropy principle."""
    return minimal_selector(data, variables, _variance, normalize=False)


def _entropy(data: np.ndarray, variable: List[str]):
    axes = tuple(x for x in range(data.ndim) if x != variable)
    return data.var(axis=axes).sum()


def entropy(data: np.ndarray, variables: List[str]):
    """Return a new VarSelector based on the minimun entropy principle."""
    return minimal_selector(data, variables, _entropy, normalize=True)


def _inclusion_entropy(data: np.ndarray, variable: List[str]):
    """H(var)"""
    axis = tuple(x for x in range(data.ndim) if x != variable)
    return stats.entropy(data.sum(axis))


def _exclusion_entropy(data: np.ndarray, variable):
    """H({variables} \ var)"""
    return stats.entropy(np.nditer(data.sum(variable)))


def _mutual_information(data, variable):
    return _inclusion_entropy(data, variable) + _exclusion_entropy(
        data, variable)


def mutual_information(data: np.ndarray, variables: List[str]):
    """Return a new VarSelector based on the Maximum mutual_information."""
    return minimal_selector(data, variables, _entropy, normalize=True)
