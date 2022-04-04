import dataclasses
import itertools
import json
import joblib
import multiprocessing
import os
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

from experiments import networks, statistics
from experiments.medical import aproximation



@dataclasses.dataclass
class Result:
    variable : str
    net : str
    error : float
    total_reduction_error : float
    partial_reduction_error: float

    @classmethod
    def from_dict(cls, dict_: dict):
        result = cls('', '', 0, 0, 0)

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


def _entropy(mata, matb):
    """Helper to check Kullback-Leibler distance."""
    mata = mata.flatten()
    matb = matb.flatten()

    loga = np.log(mata)
    logb = np.log(matb)

    result = np.multiply(mata, loga-logb)
    
    return result.sum()

def variable_aproximation(original_net, variable, error):
    """We only aproximate on the given variable"""
    cpd = original_net.get_cpds(variable)
    original_values, reduced_values, time_, modified = aproximation.aproximate_cpd(
        cpd, error)

    if modified:
        modified_cpd = cpd.copy()
        modified_cpd.values = reduced_values

        modified_net = original_net.copy()
        modified_net.add_cpds(modified_cpd)

        return modified_net
    else:
        raise ValueError('This should not happen')


def full_aproximation(original_net, error):
    """We only aproximate on the given variable"""
    modified_net = original_net.copy()
    for cpd in original_net.get_cpds():
        original_values, reduced_values, time_, modified = aproximation.aproximate_cpd(
            cpd, error, interactive=False)

        modified_cpd = cpd.copy()
        modified_cpd.values = reduced_values

        modified_net.add_cpds(modified_cpd)

    return modified_net


def prosterior_kullback_diference(original_net, modified_net, variable):
    try:
        original_posterior_values = inference.VariableElimination(
            original_net).query([variable]).values
        reduced_posterior_values = inference.VariableElimination(
            modified_net).query([variable]).values

        posterior_entropy = _entropy(original_posterior_values,
                                     reduced_posterior_values)
        return posterior_entropy
    except ValueError:
        return -1



def diference_experiment(objectives, error):
    results = statistics.Statistics()
    
    for net_name, goal_variables in objectives.items():
        original_net = networks.get(net_name)
        full_modified_net = full_aproximation(original_net, error)
        for variable in goal_variables:
            partial_modified_net = variable_aproximation(original_net, variable, error)
            
            total_reduction_error = prosterior_kullback_diference(original_net,
                                                                      full_modified_net,
                                                                      variable)
            partial_reduction_error = prosterior_kullback_diference(original_net,
                                                                        partial_modified_net,
                                                                        variable)

            results.add(Result(
                net=net_name,
                error=error,
                variable=variable,
                total_reduction_error=total_reduction_error,
                partial_reduction_error=partial_reduction_error,
            ))
            
        write_results(variable, error, net_name, results)

    return results


def parallel_diference_experiment(error):
    """Auxiliar pickable function for parallel computing.

    [1] https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
    [2] https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
    """
    return diference_experiment(OBJECTIVE_NETS, error)

def write_results(variable, error, net, results):
    result_file = f'resultados_provisionales/kullback_{net}_{variable}_{error}.json'
    with open(result_file, 'w') as file:
        file.write(results.dumps())

    results.clear()


# Configuration constants   

INTERACTIVE = True
VERBOSY = False



OBJECTIVE_NETS = {
    'hepar2.bif': ['ggtp', 'ast', 'alt', 'bilirubin'],
    'diabetes.bif': ['cho_0'],
    'munin.bif': ['L_MED_ALLCV_EW'],
    'pathfinder.bif': ['F40']
}
EXAMPLE_NETS = {'hepar2.bif': ['ggtp'],}
ERRORS = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

if __name__ == '__main__':
    
    joblib.Parallel(n_jobs=len(ERRORS))(joblib.delayed(parallel_diference_experiment)(error)
                                        for error in ERRORS)
