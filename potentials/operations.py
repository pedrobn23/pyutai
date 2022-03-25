"""
Operations module generate a new iterator to larn from it the new potentials.
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


def combine(fst, scnd):
    variables = list(set(fst.variables) | set(scnd.variables))

    fst_to_general = {var : variables.index(var) for var in fst.variables}
    scnd_to_general = {var : variables.index(var) for var in scnd.variables}

    cardinalities = dict(fst.cardinalities)
    cardinalities.update(scnd.cardinalities)


    def fst_projection(states : tuple):
        return tuple(states[fst_to_general[var]] for var in fst.variables)

    def scnd_projection(states : tuple):
        return tuple(states[scnd_to_general[var]] for var in scnd.variables)


    
    def iterable():
        for state in itertools.product(*[range(cardinalities[var]) for var in variables]):
            x_fst = fst.access(fst_projection(state))
            if x_fst == 0:
                value = 0
            else:
                y_scnd = scnd.access(scnd_projection(state))
                if y_scnd == 0:
                    value = 0
                else:
                    value = x_fst * y_scnd

            yield element.TupleElement(state, value)

    return iterable(), variables, cardinalities
    


def marginalize(fst, variable : str):
    variables = list(fst.variables)
    variables.remove(variable)

    total_to_reduced = {var : fst.variables.index(var) for var in variables}
    
    cardinalities = dict(fst.cardinalities)
    marg_var_card = cardinalities[variable]
    del cardinalities[variable]

    def aux_iter(state : Tuple[int]):
        for extra_state in range(marg_var_card):
            yield tuple(state[total_to_reduced[var]] if var in total_to_reduced else extra_state
                        for var in fst.variables)

    def iterable():
        for state in itertools.product(*[range(cardinalities[var]) for var in variables]):
            value = sum(fst.access(complete_state) for complete_state in aux_iter(state))
            yield element.TupleElement(state, value)

    return iterable(), variables, cardinalities


