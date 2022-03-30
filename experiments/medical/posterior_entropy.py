import dataclasses
import itertools
import json
import os
import statistics
import time

from typing import List

import numpy as np
import pandas as pd
import scipy

from pgmpy import inference
from pgmpy.factors.discrete import CPD

from potentials import cluster, element, indexpairs, indexmap, reductions, valuegrains
from potentials import utils as size_utils
from pyutai import trees

from experiments import networks
from experiments.medical import aproximation



@dataclasses.dataclass
class Result:
    variable : str
    net : str
    error : float
    posterior_error : float
    
    def __post_init__(self):
        if self.original_size != 0:
            self.improvement = 1 - self.reduced_size / self.original_size
        else:
            self.improvement = 0

    @classmethod
    def from_dict(cls, dict_: dict):
        result = cls('', '', 0)

        for field_ in dataclasses.fields(cls):
            try:
                setattr(result, field_.name, dict_[field_.name])
            except KeyError:
                pass
        result.__post_init__()

        return result

    def asdict(self):
        return dataclasses.asdict(self)

    def astuple(self):
        return dataclasses.astuple(self)


def _kullback(mata, matb):
    """Helper to check Kullback-Leibler distance."""
    mata = mata.flatten()
    matb = matb.flatten()
    return sum(a * (np.log(a) - np.log(b)) for a, b in zip(mata, matb))


def _entropy(orig: np.ndarray, reduc: np.ndarray) -> float:
    if sum(scipy.stats.entropy(orig, reduc)) != _kullback(orig, reduc):
        print(sum(scipy.stats.entropy(orig, reduc)), _kullback(orig, reduc))
        raise AssertionError('hola')
    return sum(scipy.stats.entropy(orig, reduc))


def variable_aproximation(original_net, variable, error):
    """We only aproximate on the given variable"""
    cpd = bayesian_net.get_cpds(variable)
    original_values, reduced_values, time_, modified = aporixmation.aproximate_cpd(
        cpd, error)

    if modified:
        modified_cpd = cpd.copy()
        modified_cpd.values = reduced_values

        modified_net = bayesian_net.copy()
        modified_net.add_cpds(modified_cpd)

        return modified_net
    else:
        raise ValueError('This should not happen')


def full_aproximation(original_net, error):
    """We only aproximate on the given variable"""
    modified_net = original_net.copy()
    for cpd in bayesian_net.get_cpds():
        original_values, reduced_values, time_, modified = aporixmation.aproximate_cpd(
            cpd, error)

        modified_cpd = cpd.copy()
        modified_cpd.values = reduced_values

        modified_net.add_cpds(modified_cpd)

    return modified_net


def prosterior_kullback_diference(original_net, modified_net):
    original_posterior_values = inference.VariableElimination(
        original_net).query([variable]).values
    reduced_posterior_values = inference.VariableElimination(
        modified_net).query([variable]).values

    posterior_entropy = _entropy(original_posterior_values,
                                 reduced_posterior_values)

    return posterior_entropy


def _posterior_kullback_diference_experiment(objectives, error):
    results = statistics.Statistics()
    
    for error in errors:
        for net_name, goal_variables in objectives.items():
            original_net = networks.get(net_name)
            for variable in goal_variables:
                modified_net = variable_aproximation(original_net, variable, error)
                posterior_error = prosterior_kullback_diference(original_net,
                                                                modified_net)

                results.add(Result(
                    net=net_name,
                    error=error,
                    variable=variable,
                    posterior_error=posterior_error,
                ))

            modified_net = full_aproximation(original_net, error)
            posterior_error = prosterior_kullback_diference(original_net,
                                                                modified_net)

            results.add(Result(
                net=net_name,
                error=error,
                variable="all",
                posterior_error=posterior_error,
            ))
                
    return results

INTERACTIVE = True
VERBOSY = False

OBJECTIVE_NETS = {
    'hepar2.bif': ['ggtp', 'ast', 'alt', 'bilirubin'],
    'diabetes.bif': ['cho_0'],
    'munin.bif': ['L_MED_ALLCV_EW'],
    'pathfinder.bif': ['F40']
}

RESULT_FILE = 'resultados_provisionales/results.json'

if __name__ == '__main__':
    errors = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    results = _prosterior_kullback_diference(OBJECTIVE_NETS, errors)
    with open(RESULT_FILE, 'w') as file:
        file.write(results.dumps())
