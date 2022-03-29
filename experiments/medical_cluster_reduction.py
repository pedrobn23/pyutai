"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
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


def ordered_elements(array: np.ndarray) -> List[element.TupleElement]:
    res = [
        element.TupleElement(state=state, value=value)
        for state, value in np.ndenumerate(array)
    ]
    res.sort(key=lambda x: x.value)
    return res


@dataclasses.dataclass
class Result:
    original_size: int
    reduced_size: int
    cls: str
    cpd: str
    error: float
    time: float
    net: str = ''
    var: str = ''
    modified: bool = True
    cardinality: int = 0

    improvement: float = dataclasses.field(init=False)

    def __post_init__(self):
        if self.original_size != 0:
            self.improvement = 1 - self.reduced_size / self.original_size
        else:
            self.improvement = 0

    @classmethod
    def from_dict(cls, dict_: dict):
        result = cls(0, 0, object, '', 0, 0)

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


def _cpd_name(cpd: CPD.TabularCPD) -> str:
    variable = cpd.variable

    conditionals = list(cpd.variables)
    conditionals.remove(variable)

    return f'CPD in {variable} conditional on {conditionals}'


def _total_cardinality(cpd: CPD.TabularCPD) -> int:
    return np.prod(cpd.cardinality)


class Statistics:

    def __init__(self):
        self.results: List[Result] = []

    @classmethod
    def from_json(cls, path):
        stats = cls()

        with open(path, 'r') as file_:
            data = file_.read()
            stats.load(data)

        return stats

    @classmethod
    def from_files(cls, *paths):
        stats = cls()

        for path in paths:
            if not path.endswith('.json'):
                raise ValueError(f'.json file expected, got: {path}.')

            with open(path) as file_:
                data = file_.read()
                stats.load(data)

        return stats

    def add(self, result):
        self.results.append(result)

    def clear(self):
        self.results.clear()

    def dumps(self) -> str:
        return json.dumps([result.asdict() for result in self.results])

    def load(self, str_: str):
        self.results += [Result.from_dict(dict_) for dict_ in json.loads(str_)]

    def dataframe(self):
        data = [result.astuple() for result in self.results]
        vars_ = [field_.name for field_ in dataclasses.fields(Result)]
        return pd.DataFrame(data, columns=vars_)

    def __add__(self, other):
        results = self.results + other.results

        ret = Statistics()
        ret.results = results

        return ret

    def __str__(self):
        return str(self.results)


INTERACTIVE = True
VERBOSY = False

class _PrunedTree:
    """Auxiliar class to implement a pruned tree creator that confort standard API."""
    def __init__(self):
        pass
    
    @classmethod
    def from_array(cls, original_values, variables):
        tree = trees.Tree.from_array(original_values, variables)
        tree.prune()

        return tree

def _aproximate_cpd(cpd : CPD.TabularCPD, error):
    """"""
    original_values = cpd.values
    ordered_elements_ = ordered_elements(original_values)
    threshold = len(np.unique(original_values))

    start = time.time()
    reduction = reductions.Reduction.from_elements(ordered_elements_,
                                                           threshold=threshold,
                                                           interactive=False)
    end = time.time()
    time_ = end - start

    n_partitions = reduction.min_partitions(error)
    if n_partitions < threshold:
        reduced_values = reduction.array(n_partitions, cpd.cardinality)
        modified = True
    else:
        reduced_values = original_values
        modified = False

    if n_partitions != len(np.unique(reduced_values)):
        raise AssertionError('This should not happen')

    if (error_ := _entropy(original_values, reduced_values)) > error:
        raise AssertionError(f'Obtained error {error_}, expected under threshold {error}')

    if INTERACTIVE:
        print(f'- original number of partitions: {threshold}')
        print(f'- reduced number of partitions: {n_partitions}')
        print(f'- goal error: {error}')
        if VERBOSY:
            print(' - Original values:\n', original_values, '\n')
            print(' - Reduced values:\n', reduced_values, '\n')


    
    return original_values, reduced_values, time_, modified

def _potential_size_test(cls : type, cpd: CPD.TabularCPD, original_values : np.ndarray, reduced_values : np.ndarray, modified):

    if not modified: # if no reduction weas carried out (both arrays are equal)
        original = cls.from_array(original_values, cpd.variables)
        reduced = original

        original_size = size_utils.total_size(original)
        reduced_size = original_size
        
    else:
        original = cls.from_array(original_values, cpd.variables)
        original_size = size_utils.total_size(original)
    
        reduced = cls.from_array(reduced_values, cpd.variables)
        reduced_size = size_utils.total_size(reduced)

    if INTERACTIVE:
        print(f'- size results for {cls} class')
        print(f'    - Original class size: {original_size}')
        print(f'    - Reduced class size: {reduced_size}')
        print(
              f'    - Total improvement: {1 - (reduced_size/original_size):.2f}% '
        )

    return original_size, reduced_size

def _size_experiment(errors):
    for cpd, net_name, net in networks.medical():
        results = Statistics()

        for error in errors:
            if INTERACTIVE:
                print(f'\n\n*** Results for {_cpd_name(cpd)} in net {net_name}. ***\n')

            original_values, reduced_values, time_, modified = _aproximate_cpd(cpd, error) 
            
            for cls in [
                    trees.Tree, _PrunedTree,
                    cluster.Cluster, valuegrains.ValueGrains,
                    indexpairs.IndexPairs, indexmap.IndexMap
            ]:

                original_size, reduced_size = _potential_size_test(cls, cpd, original_values, reduced_values, modified)
                
                results.add(
                    Result(cpd=_cpd_name(cpd),
                           cls=cls.__name__,
                           error=error,
                           original_size=original_size,
                           reduced_size=reduced_size,
                           time=time_,
                           net=net_name,
                           var=cpd.variable,
                           modified=modified,
                           cardinality=_total_cardinality(cpd),
                    ))

    return results

OBJECTIVE_NETS = {'hepar2.bif': ['ggtp', 'ast', 'alt', 'bilirubin'],
                  'diabetes.bif': ['cho_0'],
                  'munin.bif': ['L_MED_ALLCV_EW'],
                  'pathfinder.bif': ['F40']}

def _entropy(orig :np.ndarray, reduc : np.ndarray) -> float:
    return scipy.stats.entropy(orig.flatten(), reduc.flatten())

def _prosterior_kullback_diference(net, variable, error):
    for net_name, goal_variables in OBJECTIVE_NETS.items():
        bayesian_net = networks.get(net_name)

        for variable in goal_variables:
            cpd = bayesian_net.get_cpds(variable)
            original_values, reduced_values, time_, modified = _aproximate_cpd(cpd, error)

            if modified:
                modified_cpd = cpd.copy()
                modified_cpd.values = reduced_values

                modified_net = bayesian_net.copy()
                modified_net.add_cpds(modified_cpd)
            else:
                raise ValueError('This should not happen')
            

            # is left to compute posterior for both of them.
            prior_entropy = _entropy(original_values, reduced_values) 

            original_posterior_values = inference.VariableElimination(net).query([variable]).values
            reduced_posterior_values = inference.VariableElimination(modified_net).query([variable]).values

            posterior_entropy = _entropy(original_posterior_values, reduced_posterior_values)
            if INTERACTIVE:
                print(f'\n\n*** Results for {_cpd_name(cpd)} in net {net_name}. ***\n')
                print(f'   - prior error: {prior_entropy}')
                print(f'   - posterior error: {posterior_entropy}')
            
            
if __name__ == '__main__':
    errors =  [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    # for error in errors:
    #     for net, variables in OBJECTIVE_NETS.items():
    #         for variable in variables:
    #             _prosterior_kullback_diference(net, variable, error)

    size_results = _size_experiment(errors)
    filename = f'resultados_provisionales/size_results.json'
    with open(filename, 'w') as file:
        file.write(results.dumps())
